from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


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
    ap.add_argument("--source", default="focus", choices=["focus", "all"])
    ap.add_argument("--weights", default="best", choices=["best", "last"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()
    metrics_dir = meta_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    src_file = metrics_dir / f"metrics_{args.weights}_{args.source}_{args.split}.csv"
    if not src_file.exists():
        raise FileNotFoundError(
            f"No encuentro {src_file}.\n"
            f"Ejecuta antes build_metrics_best_focus_all.py --stage {stage} --split {args.split}"
        )

    df = pd.read_csv(src_file)

    metric_cols = [c for c in df.columns if c.startswith("metrics/")]
    if not metric_cols:
        raise RuntimeError("No encuentro columnas metrics/* en el CSV de entrada.")

    group_cols = ["variant", "freeze"]
    for c in group_cols:
        if c not in df.columns:
            raise RuntimeError(
                f"Falta columna '{c}' en {src_file}. Revisa el build_metrics."
            )

    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()

    out = pd.DataFrame({"variant": summary["variant"], "freeze": summary["freeze"]})
    for m in metric_cols:
        out[f"{m}_mean"] = summary[(m, "mean")]
        out[f"{m}_std"] = summary[(m, "std")]

    out_path = metrics_dir / f"metrics_summary_{args.source}_{args.split}.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] {out_path} generado")


if __name__ == "__main__":
    main()