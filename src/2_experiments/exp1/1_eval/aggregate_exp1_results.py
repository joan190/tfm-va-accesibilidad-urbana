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
    ap.add_argument(
        "--source",
        default="focus",
        choices=["focus", "all"],
        help="Usa métricas focus (solo obstaculo) o all (agregado multi-clase).",
    )
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"] / "metrics"

    src_file = meta_dir / ("metrics_last_focus.csv" if args.source == "focus" else "metrics_last_all.csv")
    if not src_file.exists():
        raise FileNotFoundError(f"No encuentro {src_file}. Ejecuta antes extract_exp1_metrics.py")

    df = pd.read_csv(src_file)
    metric_cols = [c for c in df.columns if c.startswith("metrics/")]

    summary = (
        df.groupby("variant")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    out = pd.DataFrame({"variant": summary["variant"]})
    for m in metric_cols:
        out[f"{m}_mean"] = summary[(m, "mean")]
        out[f"{m}_std"] = summary[(m, "std")]

    out_name = "metrics_summary_focus.csv" if args.source == "focus" else "metrics_summary_all.csv"
    out.to_csv(meta_dir / out_name, index=False)
    print(f"[OK] Generado {out_name}")


if __name__ == "__main__":
    main()