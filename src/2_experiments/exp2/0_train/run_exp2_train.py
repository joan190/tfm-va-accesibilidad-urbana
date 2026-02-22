from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml
from ultralytics import YOLO


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


def safe_train_overrides(cfg: dict) -> dict:
    """
    Devuelve train_overrides, pero avisa si el YAML intenta pisar parámetros que ya controlamos por run:
    epochs, batch, imgsz, seed, freeze, device, data, project, name.
    """
    overrides = cfg.get("train_overrides", {}) or {}
    forbidden = {
        "epochs",
        "batch",
        "imgsz",
        "seed",
        "freeze",
        "device",
        "data",
        "project",
        "name",
        "exist_ok",
        "verbose",
    }
    clash = sorted(set(overrides.keys()).intersection(forbidden))
    if clash:
        print(
            "\n[AVISO] train_overrides contiene claves que ya controlamos por run y se van a ignorar "
            f"si se pasan por aquí: {clash}\n"
            "Consejo: no pongas estas claves en train_overrides.\n"
        )
        for k in clash:
            overrides.pop(k, None)
    return overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine")
    ap.add_argument("--force", action="store_true", help="Reentrena aunque exista best.pt")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    stage_cfg = cfg["exp2"]["stages"][stage]

    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()
    runs_dir = (root / cfg["paths"]["runs_dir"] / stage).resolve()

    runs_index_csv = meta_dir / "runs" / "runs_index.csv"
    if not runs_index_csv.exists():
        raise FileNotFoundError(
            f"No encuentro {runs_index_csv}. Ejecuta make_yolo_dataset_exp2.py antes."
        )

    df = pd.read_csv(runs_index_csv)
    if df.empty:
        raise RuntimeError("runs_index.csv está vacío")

    train_overrides = safe_train_overrides(cfg)

    imgsz_default = int(cfg["yolo"]["imgsz"])
    device_default = str(cfg["yolo"]["device"])
    model_default = str(cfg["exp2"]["model"])

    print(f"[Exp2 Train] stage={stage} runs={len(df)}")
    print(f"[Exp2 Train] runs_dir={runs_dir}")
    print(f"[Exp2 Train] model_default={model_default}")
    print(f"[Exp2 Train] imgsz_default={imgsz_default} | device_default={device_default}")
    print(f"[Exp2 Train] train_overrides={train_overrides}")

    for i, r in df.iterrows():
        run_name = str(r["run_name"])
        dataset_yaml = Path(str(r["dataset_yaml"]))

        seed = int(r["seed"])
        freeze = int(r["freeze"])

        model_path = model_default
        epochs = int(stage_cfg["epochs"])
        batch = int(stage_cfg["batch"])
        imgsz = imgsz_default
        device = device_default

        expected_run_dir = (runs_dir / run_name).resolve()
        best_pt = expected_run_dir / "weights" / "best.pt"

        if best_pt.exists() and not args.force:
            print(f"[SKIP] {run_name} ya tiene best.pt -> {best_pt}")
            continue

        if not dataset_yaml.exists():
            raise FileNotFoundError(f"dataset_yaml no existe: {dataset_yaml}")

        print("\n" + "=" * 100)
        print(f"[RUN {i+1}/{len(df)}] {run_name}")
        print(f" stage={stage} seed={seed} freeze={freeze}")
        print(f" data={dataset_yaml}")
        print(f" model={model_path}")
        print(f" epochs={epochs} batch={batch} imgsz={imgsz} device={device}")
        print(f" out_dir={expected_run_dir}")
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
            **train_overrides,
        )

        if best_pt.exists():
            print(f"[OK] best.pt generado: {best_pt}")
        else:
            print(f"[AVISO] best.pt no encontrado tras el entrenamiento: {best_pt}")

    print("\n[OK] Entrenamiento de Exp2 terminado")


if __name__ == "__main__":
    main()