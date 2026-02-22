from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def pick_best_row(df: pd.DataFrame) -> pd.Series:
    """
    Elige la mejor época según una métrica de validación.
    Prioridad:
      1) fitness (si existe)
      2) metrics/mAP50-95 (con o sin (B))
      3) metrics/mAP50 (con o sin (B))
    """
    candidates = [
        "fitness",
        "metrics/mAP50-95",
        "metrics/mAP50-95(B)",
        "metrics/mAP50",
        "metrics/mAP50(B)",
    ]

    best_col = None
    for c in candidates:
        if c in df.columns:
            best_col = c
            break

    if best_col is None:
        raise ValueError(
            f"No encuentro columnas para elegir el mejor entre {candidates}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    s = pd.to_numeric(df[best_col], errors="coerce")
    best_idx = s.idxmax()
    return df.loc[best_idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]

    runs_index = pd.read_csv(meta_dir / "runs" / "runs_index.csv")

    rows_long = []
    rows_last = []
    rows_best = []

    for _, r in runs_index.iterrows():
        run_dir = Path(r["run_dir"])
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            continue

        df = pd.read_csv(results_csv)
        df["variant"] = r["variant"]
        df["seed"] = r["seed"]
        df["run_name"] = r["run_name"]

        rows_long.append(df)

        last = df.iloc[-1].to_dict()
        last.update(r)
        rows_last.append(last)

        best_row = pick_best_row(df).to_dict()
        best_row.update(r)
        rows_best.append(best_row)

    pd.concat(rows_long).to_csv(meta_dir / "metrics" / "metrics_long.csv", index=False)
    pd.DataFrame(rows_last).to_csv(meta_dir / "metrics" / "metrics_last.csv", index=False)
    pd.DataFrame(rows_best).to_csv(meta_dir / "metrics" / "metrics_best.csv", index=False)

    print("[OK] metrics_long.csv, metrics_last.csv y metrics_best.csv generados")


if __name__ == "__main__":
    main()