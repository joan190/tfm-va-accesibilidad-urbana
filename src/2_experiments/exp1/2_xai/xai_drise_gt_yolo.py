from __future__ import annotations

import argparse
from pathlib import Path
import random
import yaml
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def overlay_heatmap(img_bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, cm, alpha, 0)


def make_random_mask(h: int, w: int, grid: int, p: float) -> np.ndarray:
    small = (np.random.rand(grid, grid) < p).astype(np.float32)
    mask = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask, 0, 1)


def heat_entropy(heat: np.ndarray, eps: float = 1e-12) -> float:
    x = heat.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-9)
    p = x.flatten()
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / (s + eps)
    return float(-(p * np.log(p + eps)).sum())


def union_energy_ratio_in_boxes(heat: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> float:
    h, w = heat.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        mask[y1:y2, x1:x2] = 1

    total = float(heat.sum()) + 1e-9
    inside = float(heat[mask == 1].sum())
    return float(inside / total)


def pointing_game_any_box(heat: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> int:
    y, x = np.unravel_index(np.argmax(heat), heat.shape)

    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if (x1 <= x <= x2) and (y1 <= y <= y2):
            return 1
    return 0


def draw_all_gt_boxes(img_bgr: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return out


def build_gt_boxes_map(annotations_csv: Path, target_class: str) -> dict[str, list[tuple[float, float, float, float]]]:
    ann = pd.read_csv(annotations_csv)
    needed = {"image_id", "class_name", "xmin", "ymin", "xmax", "ymax"}
    miss = needed - set(ann.columns)
    if miss:
        raise ValueError(f"annotations.csv no tiene columnas necesarias: {miss}")

    target = target_class.strip().lower()
    ann["class_name_norm"] = ann["class_name"].astype(str).str.strip().str.lower()
    ann = ann[ann["class_name_norm"] == target].copy()

    gt = {}
    for img_id, g in ann.groupby("image_id", sort=False):
        boxes = []
        for _, r in g.iterrows():
            boxes.append((float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])))
        gt[str(img_id)] = boxes
    return gt


def find_gt_for_image(gt_map: dict, img_path: Path) -> list[tuple[float, float, float, float]]:
    fn = img_path.name
    st = img_path.stem
    if fn in gt_map:
        return gt_map[fn]
    if st in gt_map:
        return gt_map[st]
    return []


def drise_gt(
    yolo: YOLO,
    img_bgr: np.ndarray,
    gt_boxes: list[tuple[float, float, float, float]],
    target_class_id: int,
    n_masks: int,
    grid: int,
    p: float,
    conf: float,
    imgsz: int,
    device: str,
) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    norm = 0.0

    def score_from_result(res) -> float:
        best = 0.0
        if res.boxes is None or len(res.boxes) == 0:
            return 0.0
        for b in res.boxes:
            if int(b.cls.item()) != int(target_class_id):
                continue
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            c = float(b.conf.item())
            ious = [iou_xyxy((bx1, by1, bx2, by2), gt) for gt in gt_boxes]
            if not ious:
                continue
            s = max(ious) * c
            if s > best:
                best = s
        return best

    res0 = yolo.predict(source=img_bgr, conf=conf, verbose=False, imgsz=imgsz, device=device)[0]
    base_score = score_from_result(res0)

    for _ in range(n_masks):
        m = make_random_mask(h, w, grid=grid, p=p)
        masked = (img_bgr.astype(np.float32) * m[..., None]).astype(np.uint8)

        res = yolo.predict(source=masked, conf=conf, verbose=False, imgsz=imgsz, device=device)[0]
        s = score_from_result(res)

        if s > 0:
            acc += s * m
            norm += s

    if norm <= 0:
        return np.zeros((h, w), dtype=np.float32), float(base_score)

    heat = acc / (norm + 1e-9)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-9)
    return heat, float(base_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target_class", default="obstaculo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n_masks", type=int, default=600)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_images", type=int, default=0, help="0 = usa todas las del benchmark")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]
    annotations_csv = root / cfg["paths"]["annotations_csv"]

    xai_root = meta_dir / "xai"
    bench_csv = xai_root / "benchmark" / "benchmark_images.csv"
    if not bench_csv.exists():
        raise FileNotFoundError(f"No encuentro {bench_csv}. Ejecuta xai_make_benchmark_list.py primero.")

    bench = pd.read_csv(bench_csv)
    if bench.empty:
        raise RuntimeError("benchmark_images.csv está vacío.")

    if args.max_images and len(bench) > args.max_images:
        bench = bench.sample(args.max_images, random_state=args.seed).reset_index(drop=True)

    gt_map = build_gt_boxes_map(annotations_csv, args.target_class)

    runs_index = pd.read_csv(meta_dir / "runs" / "runs_index.csv")

    imgsz = int(cfg["yolo"]["imgsz"])
    device = str(cfg["yolo"]["device"])

    out_base = xai_root / "drise_gt"
    out_base.mkdir(parents=True, exist_ok=True)

    rows_scores = []

    for _, r in runs_index.iterrows():
        variant = r["variant"]
        seed_run = int(r["seed"])
        run_dir = Path(r["run_dir"])
        weights = run_dir / "weights" / "best.pt"
        if not weights.exists():
            continue

        yolo = YOLO(weights.as_posix())

        names = yolo.names
        target_ids = [k for k, v in names.items() if str(v).strip().lower() == args.target_class.strip().lower()]
        if not target_ids:
            print(f"[WARN] {variant} seed={seed_run}: el modelo no tiene clase '{args.target_class}' en names.")
            continue
        target_class_id = int(target_ids[0])

        out_dir = out_base / f"{variant}_seed{seed_run}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[DRISE-GT] {variant} seed={seed_run} imgs={len(bench)} masks={args.n_masks}")

        for _, b in bench.iterrows():
            img_path = Path(b["image_path"])
            img_bgr = cv2.imread(img_path.as_posix())
            if img_bgr is None:
                continue

            gt_boxes = find_gt_for_image(gt_map, img_path)
            if not gt_boxes:
                continue

            heat, base_score = drise_gt(
                yolo=yolo,
                img_bgr=img_bgr,
                gt_boxes=gt_boxes,
                target_class_id=target_class_id,
                n_masks=args.n_masks,
                grid=args.grid,
                p=args.p,
                conf=args.conf,
                imgsz=imgsz,
                device=device,
            )

            ratio_in = union_energy_ratio_in_boxes(heat, gt_boxes)
            ent = heat_entropy(heat)
            pg = pointing_game_any_box(heat, gt_boxes)

            over = overlay_heatmap(img_bgr, heat, alpha=args.alpha)
            over = draw_all_gt_boxes(over, gt_boxes)

            out_overlay = out_dir / f"{img_path.stem}_drise_gt.png"
            out_raw = out_dir / f"{img_path.stem}_drise_gt_raw.png"
            cv2.imwrite(out_overlay.as_posix(), over)
            cv2.imwrite(out_raw.as_posix(), (np.clip(heat, 0, 1) * 255).astype(np.uint8))

            rows_scores.append({
                "variant": variant,
                "seed": seed_run,
                "image_path": str(img_path),
                "target_class": args.target_class,
                "n_gt_boxes": len(gt_boxes),
                "base_score": base_score,
                "energy_ratio_in_gt": ratio_in,
                "heat_entropy": ent,
                "pointing_game": pg,
                "n_masks": args.n_masks,
                "grid": args.grid,
                "p": args.p,
                "conf": args.conf,
            })

        print(f"[OK] Outputs: {out_dir}")

    df = pd.DataFrame(rows_scores)
    out_scores = out_base / "xai_scores_gt.csv"
    df.to_csv(out_scores, index=False)

    print(f"\n[OK] Scores guardados: {out_scores}")
    if not df.empty:
        print(
            df.groupby("variant")[["energy_ratio_in_gt", "heat_entropy", "pointing_game", "base_score"]]
            .mean()
            .round(4)
        )


if __name__ == "__main__":
    main()