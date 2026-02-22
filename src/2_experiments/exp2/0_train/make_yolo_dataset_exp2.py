from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def write_yaml(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def build_variants_from_freeze_values(freeze_values: list[int]) -> list[dict]:
    variants = []
    for fr in freeze_values:
        fr = int(fr)
        variants.append({"name": f"fr{fr:02d}", "freeze": fr})
    return variants


def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp2"].get("stage", "coarse")
    stages = cfg["exp2"].get("stages", {})
    if stage not in stages:
        raise ValueError(f"Stage '{stage}' no existe en exp2.stages. Disponibles: {list(stages.keys())}")
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine (sobrescribe el YAML)")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    stage_cfg = cfg["exp2"]["stages"][stage]

    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()
    runs_dir = (root / cfg["paths"]["runs_dir"] / stage).resolve()

    meta_runs = meta_dir / "runs"
    meta_metrics = meta_dir / "metrics"
    meta_plots = meta_dir / "plots"
    meta_runs.mkdir(parents=True, exist_ok=True)
    meta_metrics.mkdir(parents=True, exist_ok=True)
    meta_plots.mkdir(parents=True, exist_ok=True)

    dataset_yaml = (root / cfg["paths"]["base_dataset_yaml"]).resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"No existe base_dataset_yaml: {dataset_yaml}")

    freeze_values = stage_cfg.get("freeze_values", None)
    if not freeze_values:
        raise ValueError(f"Falta exp2.stages.{stage}.freeze_values")

    variants_cfg = build_variants_from_freeze_values(freeze_values)

    ds_index = {v["name"]: dataset_yaml.as_posix() for v in variants_cfg}
    write_yaml(meta_runs / "datasets_index.yaml", ds_index)

    seeds = [int(s) for s in stage_cfg["seeds"]]
    epochs = int(stage_cfg["epochs"])
    batch = int(stage_cfg["batch"])
    imgsz = int(cfg["yolo"]["imgsz"])
    device = str(cfg["yolo"]["device"])
    model = str(cfg["exp2"]["model"])

    rows = []
    for v in variants_cfg:
        vname = str(v["name"])
        freeze = int(v["freeze"])

        for seed in seeds:
            run_name = f"exp2_{stage}_{vname}_seed{seed}"
            run_dir = (runs_dir / run_name).resolve()

            rows.append(
                {
                    "experiment": "exp2",
                    "stage": stage,
                    "variant": vname,
                    "seed": seed,
                    "freeze": freeze,
                    "run_name": run_name,
                    "run_dir": run_dir.as_posix(),
                    "dataset_yaml": dataset_yaml.as_posix(),
                    "model": model,
                    "epochs": epochs,
                    "batch": batch,
                    "imgsz": imgsz,
                    "device": device,
                }
            )

    df = pd.DataFrame(rows)
    out_csv = meta_runs / "runs_index.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n[OK] Exp2 ({stage}) preparado")
    print(f"[OK] dataset_yaml base: {dataset_yaml}")
    print(f"[OK] runs_index.csv: {out_csv}")
    print(f"[OK] runs_dir: {runs_dir}\n")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()