from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def draw_gt_boxes(img_bgr: np.ndarray, ann: pd.DataFrame, image_name: str, target_class: str) -> np.ndarray:
    out = img_bgr.copy()
    sub = ann[(ann["image_id"] == image_name) & (ann["class_name"] == target_class)]
    if sub.empty:
        return out

    for _, r in sub.iterrows():
        x1, y1, x2, y2 = int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            out,
            target_class,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
    return out


def put_title(img_bgr: np.ndarray, title: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    bar_h = 45
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8) + 255
    cv2.putText(bar, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return np.vstack([bar, out])


def resize_keep(img_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w == width:
        return img_bgr
    scale = width / w
    nh = int(round(h * scale))
    return cv2.resize(img_bgr, (width, nh), interpolation=cv2.INTER_LINEAR)


def resolve_list_csv(xai_root: Path, list_csv: str) -> Path:
    p = Path(list_csv)
    if p.exists():
        return p

    cand = xai_root / "cases" / p.name
    if cand.exists():
        return cand

    raise FileNotFoundError(
        f"No encuentro list_csv ni como ruta absoluta ni en xai/cases/: {list_csv}"
    )


def infer_group_from_filename(name: str) -> str:
    n = name.lower()
    if "gain" in n:
        return "gain"
    if "drop" in n:
        return "drop"
    return "misc"


def resolve_drise_overlay(drise_dir: Path, variant: str, seed: int, img_path: Path) -> Path | None:
    cand_dir = drise_dir / f"{variant}_seed{seed}"
    stem = img_path.stem
    candidates = [
        cand_dir / f"{stem}_drise_gt.png",
        cand_dir / f"{stem}-Obstacle_drise_gt.png",
        cand_dir / f"{stem}_Obstacle_drise_gt.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--list_csv", required=True, help="CSV con columna image_path (ej: cases_gain_*.csv)")

    ap.add_argument("--variants", default="v1_obstaculo,v2_obs_noobs,v3_obs_acera,v4_full")

    ap.add_argument("--paper_ab", action="store_true", help="Modo paper: solo muestra GT + A + B")
    ap.add_argument("--a", default="v1_obstaculo", help="Variante base (solo se usa con --paper_ab)")
    ap.add_argument("--b", default="v4_full", help="Variante con contexto (solo se usa con --paper_ab)")

    ap.add_argument("--seed", type=int, default=42, help="Seed a visualizar (panel comparativo)")
    ap.add_argument("--target_class", default="obstaculo")

    ap.add_argument("--max_images", type=int, default=30)
    ap.add_argument("--tile_w", type=int, default=520, help="ancho de cada columna")

    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]

    ann_path = root / cfg["paths"]["annotations_csv"]
    ann = pd.read_csv(ann_path)

    xai_root = meta_dir / "xai"
    drise_dir = xai_root / "drise_gt"

    list_path = resolve_list_csv(xai_root, args.list_csv)
    group = infer_group_from_filename(list_path.name)

    lst = pd.read_csv(list_path)
    if "image_path" not in lst.columns:
        raise RuntimeError("El CSV de entrada debe tener columna image_path.")

    image_paths = lst["image_path"].dropna().tolist()
    if args.max_images and len(image_paths) > args.max_images:
        image_paths = image_paths[: args.max_images]

    if args.paper_ab:
        variants = [args.a.strip(), args.b.strip()]
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    out_dir = xai_root / "panels_gt" / group
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Panels] list={list_path.name} group={group}")
    print(f"[Panels] n_images={len(image_paths)} variants={variants} seed={args.seed}")
    print(f"[Panels] out_dir={out_dir}")

    for imgp in image_paths:
        img_path = Path(imgp)
        if not img_path.exists():
            print(f"[WARN] No existe imagen: {img_path}")
            continue

        img_bgr = cv2.imread(img_path.as_posix())
        if img_bgr is None:
            print(f"[WARN] No puedo leer imagen: {img_path}")
            continue

        image_name = img_path.name
        left = draw_gt_boxes(img_bgr, ann, image_name, args.target_class)
        left = resize_keep(left, args.tile_w)
        left = put_title(left, f"GT ({args.target_class})")

        tiles = [left]

        for v in variants:
            overlay_path = resolve_drise_overlay(drise_dir, v, args.seed, img_path)

            if overlay_path is None:
                blank = np.zeros_like(left) + 255
                blank = put_title(blank[45:], f"{v} (missing)")
                tiles.append(blank)
                continue

            hm = cv2.imread(overlay_path.as_posix())
            if hm is None:
                blank = np.zeros_like(left) + 255
                blank = put_title(blank[45:], f"{v} (read err)")
                tiles.append(blank)
                continue

            hm = resize_keep(hm, args.tile_w)
            hm = put_title(hm, v)
            tiles.append(hm)

        max_h = max(t.shape[0] for t in tiles)
        padded = []
        for t in tiles:
            h, w = t.shape[:2]
            if h < max_h:
                pad = np.zeros((max_h - h, w, 3), dtype=np.uint8) + 255
                t = np.vstack([t, pad])
            padded.append(t)

        panel = np.hstack(padded)
        out_path = out_dir / f"{img_path.stem}_panel_seed{args.seed}.png"
        cv2.imwrite(out_path.as_posix(), panel)

    print(f"[OK] Panels en: {out_dir}")


if __name__ == "__main__":
    main()