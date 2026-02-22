from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in_csv", default="xai_scores_gt.csv")
    ap.add_argument("--out_prefix", default="xai_gt")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]

    xai_root = meta_dir / "xai"
    drise_dir = xai_root / "drise_gt"
    out_dir = xai_root / "aggregate_gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = drise_dir / args.in_csv
    if not in_path.exists():
        raise FileNotFoundError(f"No encuentro {in_path}. Ejecuta xai_drise_gt_yolo.py primero.")

    df = pd.read_csv(in_path)
    if df.empty:
        raise RuntimeError("xai_scores_gt.csv está vacío")

    by_run = (
        df.groupby(["variant", "seed"])
          .agg(
              n=("image_path", "count"),
              pointing_game=("pointing_game", "mean"),
              energy_ratio_in_gt=("energy_ratio_in_gt", "mean"),
              heat_entropy=("heat_entropy", "mean"),
              base_score=("base_score", "mean"),
          )
          .reset_index()
    )

    by_var = (
        by_run.groupby("variant")
              .agg(
                  runs=("seed", "count"),
                  n_mean=("n", "mean"),
                  pointing_game_mean=("pointing_game", "mean"),
                  pointing_game_std=("pointing_game", "std"),
                  energy_ratio_in_gt_mean=("energy_ratio_in_gt", "mean"),
                  energy_ratio_in_gt_std=("energy_ratio_in_gt", "std"),
                  heat_entropy_mean=("heat_entropy", "mean"),
                  heat_entropy_std=("heat_entropy", "std"),
                  base_score_mean=("base_score", "mean"),
                  base_score_std=("base_score", "std"),
              )
              .reset_index()
    )

    score_df = by_var.copy()
    score_df["z_point"] = zscore(score_df["pointing_game_mean"])
    score_df["z_energy"] = zscore(score_df["energy_ratio_in_gt_mean"])
    score_df["z_base"] = zscore(score_df["base_score_mean"])
    score_df["z_entropy_inv"] = -zscore(score_df["heat_entropy_mean"])

    score_df["xai_score"] = (
        0.35 * score_df["z_point"] +
        0.35 * score_df["z_energy"] +
        0.20 * score_df["z_base"] +
        0.10 * score_df["z_entropy_inv"]
    )

    score_df = score_df.sort_values("xai_score", ascending=False).reset_index(drop=True)

    out_summary = out_dir / f"{args.out_prefix}_summary.csv"
    out_run = out_dir / f"{args.out_prefix}_by_run.csv"
    out_rank = out_dir / f"{args.out_prefix}_ranking.csv"

    by_var.to_csv(out_summary, index=False)
    by_run.to_csv(out_run, index=False)
    score_df.to_csv(out_rank, index=False)

    print(f"[OK] {out_summary}")
    print(f"[OK] {out_run}")
    print(f"[OK] {out_rank}")
    print("\nTop ranking:")
    print(score_df[["variant", "xai_score", "pointing_game_mean", "energy_ratio_in_gt_mean", "heat_entropy_mean", "base_score_mean"]]
          .head(10).round(4).to_string(index=False))

if __name__ == "__main__":
    main()