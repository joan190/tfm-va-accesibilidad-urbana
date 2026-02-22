from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import yaml
from PIL import Image
from ultralytics import YOLO


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def norm(s: str) -> str:
    return str(s).strip().lower()


def find_class_id(names, target_class: str) -> Optional[int]:
    """Soporta names como list/tuple/dict."""
    t = norm(target_class)
    if isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            if norm(v) == t:
                return int(i)
        return None
    if isinstance(names, dict):
        for k, v in names.items():
            if norm(v) == t:
                return int(k)
        return None
    try:
        for i, v in enumerate(list(names)):
            if norm(v) == t:
                return int(i)
    except Exception:
        pass
    return None


def resolve_path(base: Path, maybe_rel: str) -> Path:
    p = Path(str(maybe_rel).replace("\\", "/"))
    return p if p.is_absolute() else (base / p).resolve()


def list_images(img_root: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files: List[Path] = []
    if img_root.is_file():
        return [img_root]
    for ext in exts:
        files.extend(sorted(img_root.rglob(f"*{ext}")))
    return files


def label_path_from_image(img_path: Path) -> Path:
    """
    Asume estructura Ultralytics típica: .../images/<split>/xxx.jpg -> .../labels/<split>/xxx.txt
    Si no encuentra '/images/', intenta cambiar el padre 'images'->'labels' de forma básica.
    """
    s = img_path.as_posix()
    if "/images/" in s:
        s = s.replace("/images/", "/labels/")
    lp = Path(s).with_suffix(".txt")
    return lp


def yolo_xywhn_to_xyxy(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    cx = x * W
    cy = y * H
    bw = w * W
    bh = h * H
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    # clamp
    x1 = max(0.0, min(float(W), x1))
    y1 = max(0.0, min(float(H), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    return x1, y1, x2, y2


def read_gt_boxes(label_path: Path, W: int, H: int, dataset_target_id: int) -> List[Tuple[float, float, float, float]]:
    """
    Lee labels YOLO (cls x y w h). Filtra SOLO la clase dataset_target_id.
    """
    if not label_path.exists():
        return []
    gts: List[Tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
        except Exception:
            continue
        if cid != dataset_target_id:
            continue
        x, y, w, h = map(float, parts[1:5])
        gts.append(yolo_xywhn_to_xyxy(x, y, w, h, W, H))
    return gts


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match(preds: List[Tuple[Tuple[float, float, float, float], float]], gts: List[Tuple[float, float, float, float]], iou_thr: float) -> Tuple[int, int, int]:
    """
    preds: [ (box_xyxy, conf), ... ] ya filtradas a la clase objetivo, y ordenadas por conf desc.
    gts:   [ box_xyxy, ... ]
    Devuelve: (TP, FP, FN)
    """
    matched = [False] * len(gts)
    tp = 0
    fp = 0

    for pbox, _conf in preds:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            if matched[j]:
                continue
            v = iou_xyxy(pbox, gt)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            matched[best_j] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for m in matched if not m)
    return tp, fp, fn


def resolve_split_images_from_dataset_yaml(dataset_yaml: Path, split: str) -> List[Path]:
    """
    Soporta dataset yaml estilo Ultralytics:
      path: /abs/or/rel
      train: images/train
      val: images/val
      test: images/test
    """
    ds = read_yaml(dataset_yaml)
    if split not in ds:
        raise KeyError(f"dataset_yaml no tiene split='{split}'. Keys: {list(ds.keys())}")

    base = dataset_yaml.parent
    root = base
    if "path" in ds and ds["path"] is not None:
        root = resolve_path(base, ds["path"])

    sp = ds[split]
    roots: List[Path] = []
    if isinstance(sp, (list, tuple)):
        for item in sp:
            roots.append(resolve_path(root, str(item)))
    else:
        roots.append(resolve_path(root, str(sp)))

    imgs: List[Path] = []
    for r in roots:
        imgs.extend(list_images(r))
    return imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Opcional: para resolver project_root si lo necesitas.")
    ap.add_argument("--dataset_yaml", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--target_class", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25, help="Conf threshold para predict()")
    ap.add_argument("--iou_nms", type=float, default=0.7, help="IoU NMS para predict()")
    ap.add_argument("--iou_match", type=float, default=0.5, help="IoU para contar TP vs GT")
    args = ap.parse_args()

    dataset_yaml = Path(args.dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset_yaml no existe: {dataset_yaml}")

    weights = str(args.weights)
    if weights.startswith('"') and weights.endswith('"'):
        weights = weights[1:-1]
    wpath = Path(weights)

    if args.config:
        cfg = read_yaml(Path(args.config))
        root = Path(cfg["project_root"]).resolve()
        if not wpath.is_absolute():
            wpath = (root / wpath).resolve()

    if not wpath.exists():
        raise FileNotFoundError(f"weights no existe: {wpath}")

    ds = read_yaml(dataset_yaml)
    if "names" not in ds:
        raise KeyError(f"dataset_yaml no tiene 'names': {dataset_yaml}")
    dataset_target_id = find_class_id(ds["names"], args.target_class)
    if dataset_target_id is None:
        raise ValueError(f"target_class='{args.target_class}' no está en dataset.names: {ds['names']}")

    model = YOLO(wpath.as_posix())

    model_target_id = find_class_id(model.names, args.target_class)
    if model_target_id is None:
        if isinstance(model.names, (list, tuple)) and len(model.names) == 1:
            model_target_id = 0
        else:
            raise ValueError(f"target_class='{args.target_class}' no está en model.names: {model.names}")
    images = resolve_split_images_from_dataset_yaml(dataset_yaml, args.split)
    if not images:
        raise RuntimeError(f"No encontré imágenes para split='{args.split}' en {dataset_yaml}")

    TP = FP = FN = 0
    GT_total = 0
    PRED_total = 0

    for img_path in images:
        with Image.open(img_path) as im:
            W, H = im.size

        gt_path = label_path_from_image(img_path)
        gts = read_gt_boxes(gt_path, W, H, dataset_target_id)
        GT_total += len(gts)

        res = model.predict(
            source=img_path.as_posix(),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou_nms,
            device=args.device,
            verbose=False,
        )[0]

        preds: List[Tuple[Tuple[float, float, float, float], float]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cls_id = int(b.cls.item())
                if cls_id != int(model_target_id):
                    continue
                xyxy = tuple(map(float, b.xyxy[0].tolist()))
                conf = float(b.conf.item())
                preds.append((xyxy, conf))

        preds.sort(key=lambda x: x[1], reverse=True)
        PRED_total += len(preds)

        tp, fp, fn = greedy_match(preds, gts, args.iou_match)
        TP += tp
        FP += fp
        FN += fn

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    print("\n[OK] eval_single_model_focus (1-class binary) ✅")
    print(f" dataset_yaml={dataset_yaml}")
    print(f" weights={wpath}")
    print(f" split={args.split} target_class={args.target_class}")
    print(f" dataset_target_id={dataset_target_id} | model_target_id={model_target_id}")
    print(f" conf={args.conf} iou_nms={args.iou_nms} iou_match={args.iou_match}")
    print(f" images={len(images)} | GT={GT_total} | PRED={PRED_total}")
    print(f" TP={TP} FP={FP} FN={FN}")
    print(f" Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    main()
