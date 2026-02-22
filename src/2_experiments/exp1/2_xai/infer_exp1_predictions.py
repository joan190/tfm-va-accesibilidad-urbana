from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml

from ultralytics import YOLO


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def xyxy_from_ultra(box):
    xyxy = box.xyxy[0].tolist()
    return float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max_images", type=int, default=0, help="0 = sin límite")
    ap.add_argument("--target_class", default="obstaculo")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]
    runs_dir = root / cfg["paths"]["runs_dir"]

    xai_dir = meta_dir / "xai"
    out_dir = xai_dir / "infer"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_index = pd.read_csv(meta_dir / "runs" / "runs_index.csv")

    rows = []
    targets = []

    for _, r in runs_index.iterrows():
        variant = r["variant"]
        seed = int(r["seed"])
        run_dir = Path(r["run_dir"])
        weights = run_dir / "weights" / "best.pt"
        if not weights.exists():
            print(f"[WARN] No existe best.pt: {weights}")
            continue

        dataset_yaml = Path(r["dataset_yaml"])
        if not dataset_yaml.exists():
            print(f"[WARN] No existe dataset yaml: {dataset_yaml}")
            continue

        print(f"\n[XAI Infer] {variant} seed={seed} split={args.split}")
        model = YOLO(weights.as_posix())

        dcfg = read_yaml(dataset_yaml)
        split_value = dcfg[args.split]
        images_path = Path(split_value)
        if not images_path.is_absolute():
            images_path = (dataset_yaml.parent / images_path).resolve()

        if images_path.is_dir():
            imgs = sorted([p for p in images_path.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        else:
            if images_path.suffix.lower() == ".txt":
                imgs = [Path(x.strip()) for x in images_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            else:
                raise ValueError(f"No sé interpretar split path: {images_path}")

        if args.max_images and len(imgs) > args.max_images:
            imgs = imgs[: args.max_images]

        preds = model.predict(
            source=[p.as_posix() for p in imgs],
            conf=args.conf,
            verbose=False,
            imgsz=cfg["yolo"]["imgsz"],
            device=cfg["yolo"]["device"],
        )

        for img_path, pr in zip(imgs, preds):
            if pr.boxes is None or len(pr.boxes) == 0:
                continue

            names = pr.names  # dict id->name

            for b in pr.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = xyxy_from_ultra(b)
                rows.append({
                    "variant": variant,
                    "seed": seed,
                    "image_path": str(img_path),
                    "pred_class": names.get(cls_id, str(cls_id)),
                    "pred_class_id": cls_id,
                    "conf": conf,
                    "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                })

            target_ids = [k for k, v in names.items() if str(v).strip().lower() == args.target_class.strip().lower()]
            if not target_ids:
                continue
            target_id = int(target_ids[0])

            best = None
            best_conf = -1.0
            for b in pr.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                if cls_id == target_id and conf > best_conf:
                    best_conf = conf
                    best = b

            if best is None:
                continue

            x1, y1, x2, y2 = xyxy_from_ultra(best)
            targets.append({
                "variant": variant,
                "seed": seed,
                "image_path": str(img_path),
                "target_class": args.target_class,
                "target_class_id": target_id,
                "target_conf": float(best_conf),
                "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
            })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "preds.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Guardado preds.csv: {out_csv}")

    df_t = pd.DataFrame(targets)
    out_t = out_dir / "targets.csv"
    df_t.to_csv(out_t, index=False)
    print(f"[OK] Guardado targets.csv: {out_t}")


if __name__ == "__main__":
    main()