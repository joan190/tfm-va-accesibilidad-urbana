from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml

def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in_csv", default="xai_scores_gt.csv")
    ap.add_argument("--metric", default="base_score", choices=["base_score", "energy_ratio_in_gt", "pointing_game"])
    ap.add_argument("--a", default="v1_obstaculo", help="Variante base (sin contexto)")
    ap.add_argument("--b", default="v4_full", help="Variante con contexto")
    ap.add_argument("--agg", default="mean", choices=["mean", "median"], help="Cómo agregamos seeds por imagen")
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]

    xai_root = meta_dir / "xai"
    drise_dir = xai_root / "drise_gt"
    cases_dir = xai_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    in_path = drise_dir / args.in_csv
    if not in_path.exists():
        raise FileNotFoundError(f"No encuentro {in_path}. Ejecuta xai_drise_gt_yolo.py primero.")

    df = pd.read_csv(in_path)

    df = df[df["variant"].isin([args.a, args.b])].copy()
    if df.empty:
        raise RuntimeError("No hay filas para las variantes A/B. Revisa los nombres.")

    gcols = ["variant", "image_path"]
    if args.agg == "mean":
        dfa = df.groupby(gcols, as_index=False)[args.metric].mean()
    else:
        dfa = df.groupby(gcols, as_index=False)[args.metric].median()

    piv = dfa.pivot(index="image_path", columns="variant", values=args.metric).reset_index()

    if args.a not in piv.columns or args.b not in piv.columns:
        raise RuntimeError(f"No encuentro columnas {args.a} y/o {args.b} tras el pivot. Columnas: {list(piv.columns)}")

    piv["delta"] = piv[args.b] - piv[args.a]
    piv = piv.dropna(subset=[args.a, args.b, "delta"]).copy()

    top_gain = piv.sort_values("delta", ascending=False).head(args.topk).copy()
    top_drop = piv.sort_values("delta", ascending=True).head(args.topk).copy()

    out_gain = cases_dir / f"cases_gain_{args.metric}_{args.a}_vs_{args.b}.csv"
    out_drop = cases_dir / f"cases_drop_{args.metric}_{args.a}_vs_{args.b}.csv"

    top_gain.to_csv(out_gain, index=False)
    top_drop.to_csv(out_drop, index=False)

    print(f"[OK] Casos donde {args.b} mejora vs {args.a} -> {out_gain}")
    print(f"[OK] Casos donde {args.b} empeora vs {args.a} -> {out_drop}")

    print("\nTop gains:")
    print(top_gain[["image_path", args.a, args.b, "delta"]].head(10).round(4).to_string(index=False))

    print("\nTop drops:")
    print(top_drop[["image_path", args.a, args.b, "delta"]].head(10).round(4).to_string(index=False))

if __name__ == "__main__":
    main()