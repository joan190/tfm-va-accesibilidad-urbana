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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--source",
        default="focus",
        choices=["focus", "all"],
        help="Selecciona winner usando métricas focus (solo obstaculo) o all (agregado multi-clase).",
    )
    ap.add_argument(
        "--output_prefix",
        default="winner",
        help="Prefijo de salida (winner.json, winner_ranking.csv).",
    )
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]

    # Carpeta de salida para resultados/índices
    metrics_dir = meta_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_file = meta_dir / "metrics" / ("metrics_summary_focus.csv" if args.source == "focus" else "metrics_summary_all.csv")
    if not summary_file.exists():
        raise FileNotFoundError(
            f"No encuentro {summary_file}. Ejecuta antes:\n"
            f"  - extract_exp1_metrics.py\n"
            f"  - aggregate_exp1_results.py --source {args.source}"
        )

    df = pd.read_csv(summary_file)

    winner_cfg = cfg.get("winner", {})
    mode = winner_cfg.get("mode", "weighted")
    use_means = bool(winner_cfg.get("metric_means", True))

    if mode != "weighted":
        raise ValueError("En este pipeline dejamos winner.mode=weighted para Exp1.")

    weights: dict = winner_cfg.get("weights", {})
    if not weights:
        raise ValueError("Faltan winner.weights en el YAML.")

    wsum = sum(float(v) for v in weights.values())
    if wsum <= 0:
        raise ValueError("La suma de pesos debe ser > 0.")
    weights = {k: float(v) / wsum for k, v in weights.items()}

    req = [mean_col(m) for m in weights.keys()] if use_means else list(weights.keys())
    ensure_cols(df, req, f"winner weighted score ({args.source})")

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
    df_rank = df_rank.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    top = df_rank.iloc[0].to_dict()
    winner_payload = {
        "source": args.source,
        "mode": "weighted",
        "winner_variant": top["variant"],
        "score_definition": {
            "normalized_weights": weights,
            "uses_means": use_means,
            "tie_break": tie_metrics,
        },
        "winner_score": float(top["winner_score"]),
        "winner_metrics_means": {m: float(top[mean_col(m)]) for m in weights.keys()} if use_means else {},
    }

    out_json = metrics_dir / f"{args.output_prefix}_{args.source}.json"
    out_csv = metrics_dir / f"{args.output_prefix}_{args.source}_ranking.csv"
    out_json.write_text(json.dumps(winner_payload, indent=2), encoding="utf-8")
    df_rank.to_csv(out_csv, index=False)

    print(f"\n=== Exp1 Winner ({args.source.upper()}) ===")
    print(f"Winner: {winner_payload['winner_variant']}")
    print(f"Score:  {winner_payload['winner_score']:.6f}")
    print("Weights:", winner_payload["score_definition"]["normalized_weights"])
    if tie_metrics:
        print("Tie-break:", tie_metrics)
    print(f"\n[OK] Ranking: {out_csv}")
    print(f"[OK] Winner:  {out_json}\n")

    show_cols = ["variant", "winner_score"] + [mean_col(m) for m in weights.keys()]
    for c in tie_cols:
        if c not in show_cols:
            show_cols.append(c)
    print(df_rank[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()