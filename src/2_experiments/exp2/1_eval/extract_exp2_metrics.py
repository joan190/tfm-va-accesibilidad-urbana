from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def pick_best_row(df: pd.DataFrame) -> pd.Series:
    """
    Elige la mejor época (best) con prioridad:
      1) fitness
      2) metrics/mAP50-95(B)
      3) metrics/mAP50(B)
      4) metrics/mAP50-95
      5) metrics/mAP50
    """
    candidates = [
        "fitness",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95",
        "metrics/mAP50",
    ]
    best_col = None
    for c in candidates:
        if c in df.columns:
            best_col = c
            break
    if best_col is None:
        raise ValueError(
            f"No encuentro columnas para elegir best entre {candidates}. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    best_idx = df[best_col].astype(float).idxmax()
    return df.loc[best_idx]


def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp2"].get("stage", "coarse")
    stages = cfg["exp2"].get("stages", {})
    if stage not in stages:
        raise ValueError(
            f"Stage '{stage}' no existe en exp2.stages. Disponibles: {list(stages.keys())}"
        )
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()

    runs_index = meta_dir / "runs" / "runs_index.csv"
    if not runs_index.exists():
        raise FileNotFoundError(
            f"No encuentro {runs_index}. Ejecuta make_yolo_dataset_exp2.py antes."
        )

    out_dir = meta_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_runs = pd.read_csv(runs_index)

    rows_long = []
    rows_last = []
    rows_best = []

    for _, r in df_runs.iterrows():
        run_dir = Path(r["run_dir"])
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            print(f"[AVISO] No existe results.csv: {results_csv}")
            continue

        df = pd.read_csv(results_csv)

        df["stage"] = stage
        df["variant"] = r["variant"]
        df["seed"] = int(r["seed"])
        df["run_name"] = r["run_name"]
        df["freeze"] = int(r.get("freeze", -1))

        rows_long.append(df)

        last_row = df.iloc[-1].to_dict()
        last_row.update(r.to_dict())
        last_row["stage"] = stage
        rows_last.append(last_row)

        best_row = pick_best_row(df).to_dict()
        best_row.update(r.to_dict())
        best_row["stage"] = stage
        rows_best.append(best_row)

    if not rows_long:
        raise RuntimeError("No se ha encontrado ningún results.csv en runs.")

    pd.concat(rows_long, ignore_index=True).to_csv(out_dir / "metrics_long.csv", index=False)
    pd.DataFrame(rows_last).to_csv(out_dir / "metrics_last.csv", index=False)
    pd.DataFrame(rows_best).to_csv(out_dir / "metrics_best.csv", index=False)

    print("[OK] metrics_long.csv, metrics_last.csv y metrics_best.csv generados")


if __name__ == "__main__":
    main()