from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def safe_imread(p: Path):
    return cv2.imread(p.as_posix())


def to_float_heatmap(hm_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = gray - gray.min()
    gray = gray / (gray.max() + 1e-9)
    return gray


def read_raw_if_exists(cand_overlay: Path) -> np.ndarray | None:
    raw = cand_overlay.with_name(cand_overlay.name.replace("_drise_gt.png", "_drise_gt_raw.png"))
    if raw.exists():
        raw_img = cv2.imread(raw.as_posix(), cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            return None
        hm = raw_img.astype(np.float32) / 255.0
        hm = hm - hm.min()
        hm = hm / (hm.max() + 1e-9)
        return hm
    return None


def resolve_heatmap_path(drise_dir: Path, variant: str, seed: int, img_path: Path) -> Path | None:
    base = drise_dir / f"{variant}_seed{seed}"
    cands = [
        base / f"{img_path.stem}_drise_gt.png",
        base / f"{img_path.stem}-Obstacle_drise_gt.png",
        base / f"{img_path.stem}_Obstacle_drise_gt.png",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def draw_gt_boxes(img_bgr: np.ndarray, ann: pd.DataFrame, image_name: str, target_class: str) -> np.ndarray:
    out = img_bgr.copy()
    sub = ann[(ann["image_id"] == image_name) & (ann["class_name"] == target_class)]
    if sub.empty:
        return out
    for _, r in sub.iterrows():
        x1, y1, x2, y2 = int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(out, target_class, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
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


def make_signed_diff_vis(diff01: np.ndarray) -> np.ndarray:
    d = np.clip(diff01, -1.0, 1.0)
    pos = np.clip(d, 0.0, 1.0)
    neg = np.clip(-d, 0.0, 1.0)

    vis = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    vis[..., 0] = (neg * 255).astype(np.uint8)  # azul
    vis[..., 2] = (pos * 255).astype(np.uint8)  # rojo
    vis[..., 1] = 38
    return vis


def overlay(img_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.resize(heat_bgr, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0)


def resolve_list_csv(xai_root: Path, list_csv: str) -> Path:
    p = Path(list_csv)
    if p.exists():
        return p
    cand = xai_root / "cases" / p.name
    if cand.exists():
        return cand
    raise FileNotFoundError(f"No encuentro list_csv ni como ruta absoluta ni en xai/cases/: {list_csv}")


def infer_group_from_filename(name: str) -> str:
    n = name.lower()
    if "gain" in n:
        return "gain"
    if "drop" in n:
        return "drop"
    return "misc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--list_csv", required=True, help="CSV con columna image_path (ej cases_gain_*.csv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--a", default="v1_obstaculo")
    ap.add_argument("--b", default="v4_full")
    ap.add_argument("--target_class", default="obstaculo")
    ap.add_argument("--max_images", type=int, default=30)
    ap.add_argument("--tile_w", type=int, default=520)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--save_montage", action="store_true")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]

    xai_root = meta_dir / "xai"
    drise_dir = xai_root / "drise_gt"

    ann = pd.read_csv(root / cfg["paths"]["annotations_csv"])

    list_path = resolve_list_csv(xai_root, args.list_csv)
    group = infer_group_from_filename(list_path.name)

    lst = pd.read_csv(list_path)
    if "image_path" not in lst.columns:
        raise RuntimeError("El CSV debe tener columna image_path.")

    image_paths = lst["image_path"].dropna().tolist()
    if args.max_images and len(image_paths) > args.max_images:
        image_paths = image_paths[: args.max_images]

    out_dir = xai_root / "diff_gt" / group / f"{args.b}_minus_{args.a}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Diff] group={group} b-a = {args.b} - {args.a} | seed={args.seed} | n={len(image_paths)}")
    print(f"[Diff] out_dir = {out_dir}")

    for ip in image_paths:
        img_path = Path(ip)
        if not img_path.exists():
            print(f"[WARN] No existe imagen: {img_path}")
            continue

        img_bgr = safe_imread(img_path)
        if img_bgr is None:
            continue

        pa = resolve_heatmap_path(drise_dir, args.a, args.seed, img_path)
        pb = resolve_heatmap_path(drise_dir, args.b, args.seed, img_path)
        if pa is None or pb is None:
            print(f"[WARN] No encuentro heatmaps para {img_path.name}")
            continue

        ha_overlay = safe_imread(pa)
        hb_overlay = safe_imread(pb)
        if ha_overlay is None or hb_overlay is None:
            continue

        ha = read_raw_if_exists(pa)
        hb = read_raw_if_exists(pb)
        if ha is None:
            ha = to_float_heatmap(ha_overlay)
        if hb is None:
            hb = to_float_heatmap(hb_overlay)

        ha = cv2.resize(ha, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        hb = cv2.resize(hb, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        diff = hb - ha
        m = np.max(np.abs(diff)) + 1e-9
        diffn = diff / m

        diff_vis = make_signed_diff_vis(diffn)
        diff_over = overlay(img_bgr, diff_vis, alpha=args.alpha)

        gt_img = draw_gt_boxes(img_bgr, ann, img_path.name, args.target_class)

        stem = img_path.stem
        cv2.imwrite((out_dir / f"{stem}_diff_overlay.png").as_posix(), diff_over)
        cv2.imwrite((out_dir / f"{stem}_diff_vis.png").as_posix(), diff_vis)

        if args.save_montage:
            gt_t = put_title(resize_keep(gt_img, args.tile_w), f"GT ({args.target_class})")
            a_t  = put_title(resize_keep(ha_overlay, args.tile_w), args.a)
            b_t  = put_title(resize_keep(hb_overlay, args.tile_w), args.b)
            d_t  = put_title(resize_keep(diff_over, args.tile_w), f"DIFF ({args.b} - {args.a})")

            tiles = [gt_t, a_t, b_t, d_t]
            max_h = max(t.shape[0] for t in tiles)
            padded = []
            for t in tiles:
                h, w = t.shape[:2]
                if h < max_h:
                    pad = np.zeros((max_h - h, w, 3), dtype=np.uint8) + 255
                    t = np.vstack([t, pad])
                padded.append(t)
            montage = np.hstack(padded)
            cv2.imwrite((out_dir / f"{stem}_MONTAGE.png").as_posix(), montage)

    print(f"[OK] Salidas diff en: {out_dir}")


if __name__ == "__main__":
    main()