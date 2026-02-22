from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def write_yaml(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def norm(s: str) -> str:
    return str(s).strip().lower()


def list_names(ds_cfg: dict) -> list[str]:
    names = ds_cfg.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names)


def list_model_names(model: YOLO) -> list[str]:
    """Ultralytics YOLO model.names puede ser dict {0:'acera'} o lista."""
    n = getattr(model, "names", None)
    if n is None:
        return []
    if isinstance(n, dict):
        return [n[k] for k in sorted(n.keys(), key=lambda x: int(x))]
    return list(n)


def resolve_dataset_yaml_from_config(root: Path, cfg: dict, variant: str) -> Path:
    ds_map = (cfg.get("paths", {}) or {}).get("datasets", {}) or {}
    if variant not in ds_map:
        raise KeyError(
            f"dataset_variant='{variant}' no está en paths.datasets. Keys={list(ds_map.keys())}"
        )
    p = Path(str(ds_map[variant]).replace("\\", "/"))
    return p if p.is_absolute() else (root / p).resolve()


def resolve_ds_root(ds_yaml: Path, ds_cfg: dict) -> Path:
    base = Path(str(ds_cfg.get("path", ".")).replace("\\", "/"))
    return base if base.is_absolute() else (ds_yaml.parent / base).resolve()


def find_class_id(names: list[str], target: str) -> Optional[int]:
    t = norm(target)
    for i, v in enumerate(names):
        if norm(v) == t:
            return i
    return None


def yolo_line_to_xyxy(line: str, w: int, h: int):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(float(parts[0]))
    cx = float(parts[1]) * w
    cy = float(parts[2]) * h
    bw = float(parts[3]) * w
    bh = float(parts[4]) * h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return cls, x1, y1, x2, y2


def xyxy_to_yolo_line(
    cls: int, x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> str:
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def clip_box_to_crop(x1, y1, x2, y2, cx1, cy1, cx2, cy2):
    ix1 = max(x1, cx1)
    iy1 = max(y1, cy1)
    ix2 = min(x2, cx2)
    iy2 = min(y2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_dataset_variant", required=True, help="Ej: v5_full")
    ap.add_argument("--weights", required=True, help="best.pt del ROI-only (acera)")
    ap.add_argument(
        "--out_variant",
        required=True,
        help="Ej: v5_sidewalkcrop_multiclass (debe existir en paths.datasets o se crea en meta/exp4/datasets)",
    )
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    base_yaml = resolve_dataset_yaml_from_config(root, cfg, args.base_dataset_variant)
    base_cfg = read_yaml(base_yaml)
    base_root = resolve_ds_root(base_yaml, base_cfg)
    base_names = list_names(base_cfg)

    crop_cfg = (cfg.get("sidewalk_crop", {}) or {})
    crop_class = crop_cfg.get("crop_class", "acera")

    crop_conf = float(crop_cfg.get("crop_conf", 0.25))
    crop_iou = float(crop_cfg.get("crop_iou", 0.6))
    pad = float(crop_cfg.get("pad", 0.10))

    min_crop_conf = float(crop_cfg.get("min_crop_conf", 0.5))

    keep_empty = bool(crop_cfg.get("keep_empty", True))
    min_box_px = int(crop_cfg.get("min_box_px", 4))

    out_map = (cfg.get("paths", {}) or {}).get("datasets", {}) or {}
    if args.out_variant in out_map:
        out_yaml = Path(str(out_map[args.out_variant]).replace("\\", "/"))
        out_yaml = out_yaml if out_yaml.is_absolute() else (root / out_yaml).resolve()
        out_root = out_yaml.parent
    else:
        out_root = (root / "data" / "meta" / "exp4" / "datasets" / args.out_variant).resolve()
        out_yaml = out_root / "dataset.yaml"

    out_root.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (root / weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"weights no existe: {weights}")

    device = str(cfg.get("yolo", {}).get("device", 0))

    model = YOLO(weights.as_posix())

    model_names = list_model_names(model)
    cid_crop_model = find_class_id(model_names, crop_class)
    if cid_crop_model is None:
        raise KeyError(f"No encuentro crop_class='{crop_class}' en model.names={model_names}")

    print("NOMBRES DEL MODELO:", model.names)
    print(f"Usando cid_crop_model={cid_crop_model} para crop_class='{crop_class}'")
    print(f"NOMBRES DEL DATASET BASE ({len(base_names)}): {base_names}")

    infer_imgsz = int(cfg.get("exp4", {}).get("stages", {}).get("roi_train", {}).get("imgsz", 1280))

    total_in = 0
    total_out = 0
    total_no_crop_used = 0
    total_roi_detected = 0
    total_roi_used = 0

    for split in ["train", "val", "test"]:
        if split not in base_cfg:
            continue

        img_rel = Path(str(base_cfg[split]).replace("\\", "/"))
        img_dir = img_rel if img_rel.is_absolute() else (base_root / img_rel).resolve()

        lab_dir = (base_root / "labels" / split).resolve()
        if not lab_dir.exists():
            lab_dir = Path(str(img_dir).replace("images", "labels")).resolve()

        out_img_dir = out_root / "images" / split
        out_lab_dir = out_root / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lab_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            print(f"[AVISO] split={split}: no existe el directorio de imágenes {img_dir}, se omite")
            continue

        imgs = sorted(
            [p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        )

        for im_path in imgs:
            total_in += 1
            img = cv2.imread(im_path.as_posix())
            if img is None:
                continue
            h, w = img.shape[:2]

            pred = model.predict(
                source=im_path.as_posix(),
                imgsz=infer_imgsz,
                conf=crop_conf,
                iou=crop_iou,
                device=device,
                verbose=False,
                classes=[cid_crop_model],
                max_det=10,
            )

            crop_box = None

            if pred and len(pred) > 0 and pred[0].boxes is not None and len(pred[0].boxes) > 0:
                total_roi_detected += 1
                boxes = pred[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                best_i = int(np.argmax(confs))
                best_conf = float(confs[best_i])

                if best_conf >= min_crop_conf:
                    x1, y1, x2, y2 = xyxy[best_i].tolist()

                    bw = max(x2 - x1, 1.0)
                    bh = max(y2 - y1, 1.0)
                    px = pad * bw
                    py = pad * bh

                    cx1 = max(0, int(np.floor(x1 - px)))
                    cy1 = max(0, int(np.floor(y1 - py)))
                    cx2 = min(w, int(np.ceil(x2 + px)))
                    cy2 = min(h, int(np.ceil(y2 + py)))

                    if cx2 > cx1 and cy2 > cy1:
                        crop_box = (cx1, cy1, cx2, cy2)
                        total_roi_used += 1

            if crop_box is None:
                crop_box = (0, 0, w, h)
                total_no_crop_used += 1

            cx1, cy1, cx2, cy2 = crop_box
            crop = img[cy1:cy2, cx1:cx2].copy()
            ch, cw = crop.shape[:2]
            if ch <= 0 or cw <= 0:
                continue

            src_lab = lab_dir / f"{im_path.stem}.txt"
            out_lines = []

            if src_lab.exists():
                for line in src_lab.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parsed = yolo_line_to_xyxy(line, w, h)
                    if parsed is None:
                        continue
                    cls, x1, y1, x2, y2 = parsed

                    clipped = clip_box_to_crop(x1, y1, x2, y2, cx1, cy1, cx2, cy2)
                    if clipped is None:
                        continue
                    ix1, iy1, ix2, iy2 = clipped

                    ix1 -= cx1
                    ix2 -= cx1
                    iy1 -= cy1
                    iy2 -= cy1

                    if (ix2 - ix1) < min_box_px or (iy2 - iy1) < min_box_px:
                        continue

                    out_lines.append(xyxy_to_yolo_line(int(cls), ix1, iy1, ix2, iy2, cw, ch))

            if (not out_lines) and (not keep_empty):
                continue

            out_img_path = out_img_dir / im_path.name
            out_lab_path = out_lab_dir / f"{im_path.stem}.txt"

            cv2.imwrite(out_img_path.as_posix(), crop)
            out_lab_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

            total_out += 1

    out_cfg = {
        "path": out_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": base_names,
    }
    write_yaml(out_yaml, out_cfg)

    print("[OK] Dataset Sidewalk-crop MULTICLASE creado")
    print(f" base: {base_yaml}")
    print(f" out:  {out_yaml}")
    print(f" crop_class='{crop_class}' (id en el modelo={cid_crop_model})")
    print(f" weights={weights}")
    print(f" infer_conf_filter(crop_conf)={crop_conf} | min_crop_conf_for_cropping={min_crop_conf}")
    print(f" kept {total_out}/{total_in} images (keep_empty={keep_empty})")
    print(f" ROI detectado en {total_roi_detected}/{total_in} imágenes (tras conf={crop_conf})")
    print(f" ROI usado para recortar en {total_roi_used}/{total_in} imágenes (best_conf >= {min_crop_conf})")
    print(f" fallback a ORIGINAL (sin recorte) en {total_no_crop_used}/{total_in} imágenes")


if __name__ == "__main__":
    main()