from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def resolve_split_images(dataset_yaml: Path, split: str) -> list[Path]:
    dcfg = read_yaml(dataset_yaml)
    split_value = dcfg.get(split)
    if split_value is None:
        raise KeyError(f"dataset.yaml no tiene clave '{split}'")

    p = Path(str(split_value).replace("\\", "/"))
    if not p.is_absolute():
        p = (dataset_yaml.parent / p).resolve()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if p.is_dir():
        imgs = [x for x in p.rglob("*") if x.suffix.lower() in exts]
        return sorted(imgs)

    if p.suffix.lower() == ".txt":
        base = p.parent
        lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        out = []
        for ln in lines:
            ln = ln.replace("\\", "/")
            q = Path(ln)
            if not q.is_absolute():
                q = (base / q).resolve()
            out.append(q)
        return out

    raise ValueError(f"No sé interpretar split path: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_class", default="obstaculo")
    ap.add_argument("--pos_frac", type=float, default=0.6, help="fracción de imágenes con obstáculo (0-1)")
    ap.add_argument("--use_stratified", action="store_true", help="estratifica usando annotations.csv (GT)")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]
    annotations_csv = root / cfg["paths"]["annotations_csv"]

    xai_root = meta_dir / "xai"
    bench_dir = xai_root / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    datasets_index = read_yaml(meta_dir / "datasets_index.yaml")
    any_dataset_yaml = Path(next(iter(datasets_index.values())))
    imgs = resolve_split_images(any_dataset_yaml, args.split)

    if not imgs:
        raise RuntimeError(f"No encontré imágenes en split={args.split} usando {any_dataset_yaml}")

    df_imgs = pd.DataFrame({"image_path": [str(p.resolve()) for p in imgs]})

    if not args.use_stratified:
        bench = df_imgs.sample(n=min(args.n, len(df_imgs)), random_state=args.seed).copy()
        bench["has_obstacle_gt"] = pd.NA
    else:
        ann = pd.read_csv(annotations_csv)
        if "image_id" not in ann.columns or "class_name" not in ann.columns:
            raise ValueError("annotations.csv debe tener columnas: image_id, class_name (y bbox).")

        target = args.target_class.strip().lower()
        ann["class_name_norm"] = ann["class_name"].astype(str).str.strip().str.lower()

        df_imgs["filename"] = df_imgs["image_path"].apply(lambda x: Path(x).name)

        pos_set = set(ann.loc[ann["class_name_norm"] == target, "image_id"].astype(str))
        df_imgs["stem"] = df_imgs["image_path"].apply(lambda x: Path(x).stem)

        df_imgs["has_obstacle_gt"] = df_imgs.apply(
            lambda r: (r["filename"] in pos_set) or (r["stem"] in pos_set),
            axis=1
        )

        pos = df_imgs[df_imgs["has_obstacle_gt"] == True]
        neg = df_imgs[df_imgs["has_obstacle_gt"] == False]

        n_total = min(args.n, len(df_imgs))
        n_pos = int(round(n_total * args.pos_frac))
        n_neg = n_total - n_pos

        if len(pos) < n_pos:
            n_pos = len(pos)
            n_neg = n_total - n_pos
        if len(neg) < n_neg:
            n_neg = len(neg)
            n_pos = n_total - n_neg

        bench = pd.concat([
            pos.sample(n=n_pos, random_state=args.seed) if n_pos > 0 else pos.head(0),
            neg.sample(n=n_neg, random_state=args.seed) if n_neg > 0 else neg.head(0),
        ], axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    bench["split"] = args.split
    out_csv = bench_dir / "benchmark_images.csv"
    bench.to_csv(out_csv, index=False)
    print(f"[OK] Benchmark guardado: {out_csv}")
    print(bench.head(10).to_string(index=False))


if __name__ == "__main__":
    main()