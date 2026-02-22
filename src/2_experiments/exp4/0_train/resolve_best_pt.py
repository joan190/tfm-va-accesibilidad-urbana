from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def conf_tag(conf: float | None) -> str:
    if conf is None:
        return "def"
    return f"conf{int(round(conf * 100)):02d}"


def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp4"].get("stage", "coarse")
    stages = cfg["exp4"].get("stages", {})
    if stage not in stages:
        raise ValueError(f"Stage '{stage}' no existe en exp4.stages. Disponibles: {list(stages.keys())}")
    return stage


def pick_best_seed_row(df: pd.DataFrame, winner_variant: str, weights: dict[str, float]) -> pd.Series:
    sub = df[df["variant"] == winner_variant].copy()
    if sub.empty:
        raise RuntimeError(f"No encuentro filas para variant='{winner_variant}' en metrics_focus.")

    score = 0.0
    used = 0
    for m, w in weights.items():
        if m in sub.columns:
            score = score + float(w) * sub[m].astype(float)
            used += 1

    if used > 0:
        sub["__score"] = score
        sub = sub.sort_values("__score", ascending=False)
        return sub.iloc[0]

    pref = [c for c in ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/recall(B)", "metrics/precision(B)"] if c in sub.columns]
    if pref:
        sub = sub.sort_values(pref, ascending=[False] * len(pref))
    return sub.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--source", default="focus", choices=["focus", "all"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--weights", default="best", choices=["best", "last"])
    ap.add_argument("--conf", type=float, default=None)
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    stage = get_stage(cfg, args.stage)

    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()
    metrics_dir = meta_dir / "metrics"
    tag = conf_tag(args.conf)

    winner_candidates = [
        metrics_dir / f"winner_{args.source}_{args.split}_{tag}.json",
        metrics_dir / f"winner_{args.source}_{args.split}.json",
    ]
    winner_json = None
    for c in winner_candidates:
        if c.exists():
            winner_json = c
            break

    winner_variant = None
    if winner_json:
        payload = json.loads(winner_json.read_text(encoding="utf-8"))
        winner_variant = payload.get("winner_variant", None)

    if winner_variant is None:
        rank_candidates = [
            metrics_dir / f"winner_{args.source}_{args.split}_{tag}_ranking.csv",
            metrics_dir / f"winner_{args.source}_{args.split}_ranking.csv",
        ]
        rank_csv = None
        for c in rank_candidates:
            if c.exists():
                rank_csv = c
                break
        if rank_csv is None:
            raise FileNotFoundError(f"No encuentro winner json ni ranking en {metrics_dir}")

        df_rank = pd.read_csv(rank_csv)
        if df_rank.empty or "variant" not in df_rank.columns:
            raise RuntimeError(f"Ranking inválido: {rank_csv}")
        winner_variant = str(df_rank.iloc[0]["variant"])

    src = metrics_dir / f"metrics_{args.weights}_{args.source}_{args.split}_{tag}.csv"
    if not src.exists():
        src = metrics_dir / f"metrics_{args.weights}_{args.source}_{args.split}.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"No encuentro metrics csv para resolver best.pt.\n"
            f"Esperaba: metrics_{args.weights}_{args.source}_{args.split}_{tag}.csv"
        )

    df = pd.read_csv(src)
    winner_cfg = cfg.get("winner", {}) or {}
    weights_map = winner_cfg.get("weights", {}) or {}
    wsum = sum(float(v) for v in weights_map.values()) or 1.0
    weights_map = {k: float(v) / wsum for k, v in weights_map.items()}

    best_row = pick_best_seed_row(df, winner_variant, weights_map)
    run_dir = Path(str(best_row["run_dir"]))
    best_pt = run_dir / "weights" / f"{args.weights}.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"No existe {best_pt}")

    print(best_pt.as_posix())


if __name__ == "__main__":
    main()
