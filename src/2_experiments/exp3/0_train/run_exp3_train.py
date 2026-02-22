from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml
from ultralytics import YOLO


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp3"].get("stage", "coarse")
    stages = cfg["exp3"].get("stages", {})
    if stage not in stages:
        raise ValueError(f"Stage '{stage}' no existe en exp3.stages. Disponibles: {list(stages.keys())}")
    return stage


def safe_train_overrides(cfg: dict) -> dict:
    overrides = cfg.get("train_overrides", {}) or {}
    forbidden = {"epochs", "batch", "imgsz", "seed", "freeze", "device", "data", "project", "name", "exist_ok", "verbose"}
    for k in list(overrides.keys()):
        if k in forbidden:
            overrides.pop(k, None)
    return overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine")
    ap.add_argument("--force", action="store_true", help="Re-entrena aunque exista best.pt")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    stage_cfg = cfg["exp3"]["stages"][stage]

    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()
    runs_dir = (root / cfg["paths"]["runs_dir"] / stage).resolve()

    runs_index_csv = meta_dir / "runs" / "runs_index.csv"
    if not runs_index_csv.exists():
        raise FileNotFoundError(f"No encuentro {runs_index_csv}. Ejecuta make_yolo_dataset_exp3.py antes.")

    df = pd.read_csv(runs_index_csv)
    if df.empty:
        raise RuntimeError("runs_index.csv está vacío")

    dataset_yaml = None
    imgsz = int(cfg["yolo"]["imgsz"])
    device = str(cfg["yolo"]["device"])
    model_path = str(cfg["exp3"]["model"])
    freeze = int(cfg["exp3"]["freeze"])
    epochs = int(stage_cfg["epochs"])
    batch = int(stage_cfg["batch"])

    train_overrides = safe_train_overrides(cfg)
    aug_map: dict = cfg.get("augmentations", {}) or {}

    print(f"[Exp3 Train] stage={stage} runs={len(df)} | runs_dir={runs_dir}")
    print(f"[Exp3 Train] model={model_path} freeze={freeze} imgsz={imgsz} epochs={epochs} batch={batch} device={device}")
    print(f"[Exp3 Train] train_overrides={train_overrides}")

    for i, r in df.iterrows():
        variant = str(r["variant"]) 
        seed = int(r["seed"])
        run_name = str(r["run_name"])
        run_dir = Path(str(r["run_dir"])) 
        dataset_yaml = Path(str(r["dataset_yaml"]))

        if not dataset_yaml.exists():
            raise FileNotFoundError(f"dataset_yaml no existe: {dataset_yaml}")

        expected_run_dir = (runs_dir / run_name).resolve()
        best_pt = expected_run_dir / "weights" / "best.pt"
        if best_pt.exists() and not args.force:
            print(f"[SKIP] {run_name} ya tiene best.pt -> {best_pt}")
            continue

        aug_overrides = aug_map.get(variant, {}) or {}
        if not aug_overrides:
            print(f"[WARN] No encuentro augmentations para variant='{variant}'. Usaré solo train_overrides (sin DA específica).")

        merged_overrides = {}
        merged_overrides.update(train_overrides)
        merged_overrides.update(aug_overrides)

        print("\n" + "=" * 100)
        print(f"[RUN {i+1}/{len(df)}] {run_name}")
        print(f" variant(DA)={variant} seed={seed} freeze={freeze}")
        print(f" data={dataset_yaml}")
        print(f" out_dir={expected_run_dir}")
        print(f" merged_overrides(keys)={sorted(list(merged_overrides.keys()))}")
        print("=" * 100)

        model = YOLO(model_path)
        model.train(
            data=dataset_yaml.as_posix(),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            seed=seed,
            freeze=freeze,
            project=runs_dir.as_posix(),
            name=run_name,
            exist_ok=True,
            verbose=True,
            **merged_overrides,
        )

        if best_pt.exists():
            print(f"[OK] best.pt generado: {best_pt}")
        else:
            print(f"[WARN] best.pt no encontrado tras train: {best_pt}")

    print("\n[OK] Exp3 training terminado")


if __name__ == "__main__":
    main()
