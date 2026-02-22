from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def write_yaml(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp3"].get("stage", "coarse")
    stages = cfg["exp3"].get("stages", {})
    if stage not in stages:
        raise ValueError(f"Stage '{stage}' no existe en exp3.stages. Disponibles: {list(stages.keys())}")
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    stage_cfg = cfg["exp3"]["stages"][stage]

    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()
    runs_dir = (root / cfg["paths"]["runs_dir"] / stage).resolve()

    (meta_dir / "runs").mkdir(parents=True, exist_ok=True)
    (meta_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (meta_dir / "plots").mkdir(parents=True, exist_ok=True)

    dataset_yaml = (root / cfg["paths"]["base_dataset_yaml"]).resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"base_dataset_yaml no existe: {dataset_yaml}")

    model = str(cfg["exp3"]["model"])
    freeze = int(cfg["exp3"]["freeze"])
    epochs = int(stage_cfg["epochs"])
    batch = int(stage_cfg["batch"])
    imgsz = int(cfg["yolo"]["imgsz"])
    device = str(cfg["yolo"]["device"])

    variants = stage_cfg["variants"]
    seeds = [int(s) for s in stage_cfg["seeds"]]

    ds_index = {v: dataset_yaml.as_posix() for v in variants}
    write_yaml(meta_dir / "runs" / "datasets_index.yaml", ds_index)

    rows = []
    for v in variants:
        for seed in seeds:
            run_name = f"exp3_{stage}_{v}_seed{seed}"
            run_dir = (runs_dir / run_name).resolve()
            rows.append(
                {
                    "experiment": "exp3",
                    "stage": stage,
                    "variant": v, 
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
    out_csv = meta_dir / "runs" / "runs_index.csv"
    df.to_csv(out_csv, index=False)

    print("[OK] Exp3 preparado")
    print(f"[OK] stage={stage}")
    print(f"[OK] dataset_yaml: {dataset_yaml}")
    print(f"[OK] runs_index.csv: {out_csv}")
    print(f"[OK] runs_dir (ultralytics): {runs_dir}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
