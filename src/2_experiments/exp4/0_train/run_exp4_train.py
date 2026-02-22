from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import pandas as pd
import yaml
from ultralytics import YOLO
import torch
from ultralytics.engine.trainer import BaseTrainer

def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp4"].get("stage", "coarse")
    stages = cfg["exp4"].get("stages", {})
    if stage not in stages:
        raise ValueError(f"Stage '{stage}' no existe en exp4.stages. Disponibles: {list(stages.keys())}")
    return stage


def safe_overrides(cfg: dict) -> dict:
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
        "optimizer",
        "lr0",
        "model",
    }
    clean = dict(overrides)
    for k in list(clean.keys()):
        if k in forbidden:
            clean.pop(k, None)
    return clean


def expand_env_tokens(s: str) -> str:
    if s is None:
        return s
    s = str(s).strip().strip('"').strip("'")

    s2 = os.path.expandvars(s)

    pattern = r"\$(\w+)|\$\{(\w+)\}"

    def repl(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        return os.environ.get(name, m.group(0))

    s2 = re.sub(pattern, repl, s2)
    return s2


def looks_like_path(s: str) -> bool:
    s = str(s)
    return ("/" in s) or ("\\" in s) or s.lower().endswith((".pt", ".pth"))


def resolve_model_path(root: Path, model_init_raw: str, run_name: str) -> str:
    model_init = expand_env_tokens(model_init_raw)

    if re.search(r"\$\w+|\$\{\w+\}|%\w+%", model_init):
        raise ValueError(
            f"[{run_name}] model_init contiene variables sin resolver: '{model_init_raw}' -> '{model_init}'.\n"
            f"Solución:\n"
            f"  - En CMD: usa %BASE_BEST_PT% o ejecuta 'set BASE_BEST_PT=...'\n"
            f"  - O pasa una ruta absoluta real al --model"
        )

    if looks_like_path(model_init):
        mp = Path(model_init)
        if not mp.is_absolute():
            mp = (root / mp).resolve()
        model_init = mp.as_posix()
        if not Path(model_init).exists():
            raise FileNotFoundError(f"[{run_name}] model_init no existe: {model_init}")
        return model_init

    return str(model_init)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine | roi_train | crop_coarse | crop_fine | hn_fine ...")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()
    runs_dir = (root / cfg["paths"]["runs_dir"] / stage).resolve()

    runs_index_csv = meta_dir / "runs" / "runs_index.csv"
    if not runs_index_csv.exists():
        raise FileNotFoundError(f"No encuentro {runs_index_csv}. Ejecuta make_yolo_dataset_exp4.py antes.")

    df = pd.read_csv(runs_index_csv)
    if df.empty:
        raise RuntimeError("runs_index.csv está vacío")

    device_default = str(cfg.get("yolo", {}).get("device", 0))
    model_default = str(cfg["exp4"]["model"])
    freeze_default = int(cfg["exp4"]["freeze"])

    base_overrides = safe_overrides(cfg)
    aug_map = cfg.get("augmentations", {}) or {}
    da_policy_default = "da_baseline"

    print(f"[Exp4 Train] stage={stage} runs={len(df)} | runs_dir={runs_dir}")
    print(f"[Exp4 Train] defaults: model={model_default} freeze={freeze_default} device={device_default}")
    print(f"[Exp4 Train] base_overrides={base_overrides}")
    print("[Exp4 Train] Parche Lion activado (optimizer='Lion')")

    for i, r in df.iterrows():
        run_name = str(r["run_name"])
        dataset_yaml = Path(str(r["dataset_yaml"]))
        dataset_variant = str(r.get("dataset_variant", ""))

        seed = int(r["seed"])
        epochs = int(r["epochs"])
        batch = int(r["batch"])
        imgsz = int(r["imgsz"])
        lr0 = float(r["lr0"])
        optimizer = str(r["optimizer"])
        da_policy = str(r.get("da_policy", da_policy_default))

        weight_decay = r.get("weight_decay", None)
        warmup_epochs = r.get("warmup_epochs", None)

        model_init_raw = str(r.get("model", model_default))
        freeze = int(r.get("freeze", freeze_default))
        device = str(r.get("device", device_default))

        expected_run_dir = (runs_dir / run_name).resolve()
        best_pt = expected_run_dir / "weights" / "best.pt"
        if best_pt.exists() and not args.force:
            print(f"[SKIP] {run_name} ya tiene best.pt")
            continue

        if not dataset_yaml.exists():
            raise FileNotFoundError(f"[{run_name}] dataset_yaml no existe: {dataset_yaml}")

        model_init = resolve_model_path(root, model_init_raw, run_name)

        aug_overrides = aug_map.get(da_policy, {}) or {}
        if not aug_overrides:
            print(f"[AVISO] No encuentro augmentations para '{da_policy}'. Entrenaré sin DA específica.")

        merged = {}
        merged.update(base_overrides)
        merged.update(aug_overrides)

        if weight_decay is not None and pd.notna(weight_decay):
            merged["weight_decay"] = float(weight_decay)
        if warmup_epochs is not None and pd.notna(warmup_epochs):
            merged["warmup_epochs"] = int(warmup_epochs)

        print("\n" + "=" * 120)
        print(f"[RUN {i+1}/{len(df)}] {run_name}")
        print(f" dataset_variant={dataset_variant}")
        print(f" model_init={model_init}")
        print(
            f" seed={seed} imgsz={imgsz} batch={batch} lr0={lr0} optimizer={optimizer} "
            f"freeze={freeze} da={da_policy} device={device}"
        )
        if "weight_decay" in merged or "warmup_epochs" in merged:
            print(
                f" extra: weight_decay={merged.get('weight_decay', 'default')} "
                f"warmup_epochs={merged.get('warmup_epochs', 'default')}"
            )
        print(f" data={dataset_yaml}")
        print(f" out_dir={expected_run_dir}")
        print("=" * 120)

        model = YOLO(model_init)
        model.train(
            data=dataset_yaml.as_posix(),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            seed=seed,
            freeze=freeze,
            optimizer=optimizer,
            lr0=lr0,
            project=runs_dir.as_posix(),
            name=run_name,
            exist_ok=True,
            verbose=True,
            **merged,
        )

    print("\n[OK] Entrenamiento Exp4 terminado")


if __name__ == "__main__":
    main()