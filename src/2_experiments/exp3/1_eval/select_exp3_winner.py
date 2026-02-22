from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def mean_col(metric: str) -> str:
    return f"{metric}_mean"


def ensure_cols(df: pd.DataFrame, cols: list[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas para {context}: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )


def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp3"].get("stage", "coarse")
    stages = cfg["exp3"].get("stages", {})
    if stage not in stages:
        raise ValueError(
            f"Stage '{stage}' no existe en exp3.stages. Disponibles: {list(stages.keys())}"
        )
    return stage


def conf_tag(conf: float | None) -> str:
    if conf is None:
        return "def"
    return f"conf{int(round(conf * 100)):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine")
    ap.add_argument("--source", default="focus", choices=["focus", "all"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--output_prefix", default="winner")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()
    metrics_dir = meta_dir / "metrics"

    tag = conf_tag(args.conf)
    summary_file = metrics_dir / f"metrics_summary_{args.source}_{args.split}_{tag}.csv"
    if not summary_file.exists():
        raise FileNotFoundError(
            f"No encuentro {summary_file}. Ejecuta antes aggregate_exp3_results.py"
        )

    df = pd.read_csv(summary_file)

    winner_cfg = cfg.get("winner", {}) or {}
    mode = winner_cfg.get("mode", "weighted")
    use_means = bool(winner_cfg.get("metric_means", True))
    if mode != "weighted":
        raise ValueError("winner.mode debe ser 'weighted'")

    weights: dict = winner_cfg.get("weights", {}) or {}
    if not weights:
        raise ValueError("Faltan winner.weights en el YAML.")

    wsum = sum(float(v) for v in weights.values())
    if wsum <= 0:
        raise ValueError("La suma de pesos debe ser > 0.")
    weights = {k: float(v) / wsum for k, v in weights.items()}

    req = [mean_col(m) for m in weights.keys()] if use_means else list(weights.keys())
    ensure_cols(df, req, f"winner weighted score ({args.source}/{args.split}/{tag})")

    df_rank = df.copy()
    score = 0.0
    for m, w in weights.items():
        col = mean_col(m) if use_means else m
        score = score + w * df_rank[col]
    df_rank["winner_score"] = score

    tie_metrics = winner_cfg.get("tie_break", []) or []
    tie_cols = [mean_col(tm) if use_means else tm for tm in tie_metrics]
    if tie_cols:
        ensure_cols(df_rank, tie_cols, "tie_break")

    sort_cols = ["winner_score"] + tie_cols
    df_rank = df_rank.sort_values(
        by=sort_cols, ascending=[False] * len(sort_cols)
    ).reset_index(drop=True)

    top = df_rank.iloc[0].to_dict()

    winner_payload = {
        "stage": stage,
        "source": args.source,
        "split": args.split,
        "conf": args.conf if args.conf is not None else "default",
        "mode": "weighted",
        "winner_variant": top["variant"],
        "winner_score": float(top["winner_score"]),
        "score_definition": {
            "normalized_weights": weights,
            "uses_means": use_means,
            "tie_break": tie_metrics,
        },
        "winner_metrics_means": {m: float(top[mean_col(m)]) for m in weights.keys()} if use_means else {},
    }

    out_json = metrics_dir / f"{args.output_prefix}_{args.source}_{args.split}_{tag}.json"
    out_csv = metrics_dir / f"{args.output_prefix}_{args.source}_{args.split}_{tag}_ranking.csv"

    out_json.write_text(json.dumps(winner_payload, indent=2), encoding="utf-8")
    df_rank.to_csv(out_csv, index=False)

    print(f"\n=== Exp3 Ganador ({stage} / {args.source.upper()} / {args.split} / {tag}) ===")
    print(f"Ganador: {winner_payload['winner_variant']}")
    print(f"Puntuación: {winner_payload['winner_score']:.6f}")
    print("Pesos:", winner_payload["score_definition"]["normalized_weights"])
    if tie_metrics:
        print("Desempate:", tie_metrics)

    print(f"\n[OK] Ranking: {out_csv}")
    print(f"[OK] Ganador:  {out_json}\n")

    show_cols = ["variant", "winner_score"] + [mean_col(m) for m in weights.keys()] + tie_cols
    show_cols = [c for c in show_cols if c in df_rank.columns]
    print(df_rank[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()