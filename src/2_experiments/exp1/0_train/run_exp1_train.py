from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import yaml
import pandas as pd


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def get_class_id(dataset_yaml_path: Path, class_name: str) -> int:
    ds = read_yaml(dataset_yaml_path)
    names = ds.get("names", {})
    # names puede ser {0:"obstaculo"} o {"0":"obstaculo"}
    for k, v in names.items():
        if str(v).strip().lower() == class_name.strip().lower():
            return int(k)
    raise ValueError(f"No encuentro '{class_name}' en names de {dataset_yaml_path}. names={names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    meta_dir = root / cfg["paths"]["meta_dir"]
    runs_dir = root / cfg["paths"]["runs_dir"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    (meta_dir / "runs").mkdir(parents=True, exist_ok=True)

    datasets_index = read_yaml(meta_dir / "datasets_index.yaml")
    y = cfg["yolo"]
    seeds = cfg.get("seeds", [y.get("seed", 42)])

    focus_cfg = cfg.get("focus", {})
    focus_class_name = focus_cfg.get("class_name", "obstaculo")

    workers = int(y.get("workers", 0))
    close_mosaic = int(y.get("close_mosaic", 10))

    OPTIONAL_KEYS = [
        "momentum", "weight_decay", "cos_lr", "warmup_epochs",
        "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "perspective", "shear",
        "mosaic", "mixup", "close_mosaic", "copy_paste", "cutmix",
        "fliplr", "flipud",
        "erasing", "auto_augment",
    ]

    runs_index = []

    for variant, dataset_yaml in datasets_index.items():
        dataset_yaml_path = Path(dataset_yaml)
        focus_class_id = get_class_id(dataset_yaml_path, focus_class_name)

        for seed in seeds:
            run_name = f"{variant}_seed{seed}"
            run_dir = (runs_dir / run_name).resolve()

            print(f"\n[Exp1] Variant={variant} | Seed={seed}")
            if str(y.get("device", "cpu")).lower() == "cpu":
                print("[Device] CPU")
            else:
                print(f"[Device] GPU CUDA:{y.get('device')}")

            best_pt_existing = run_dir / "weights" / "best.pt"
            last_pt_existing = run_dir / "weights" / "last.pt"

            if best_pt_existing.exists() or last_pt_existing.exists():
                print(f"[SKIP TRAIN] Ya existe un entrenamiento en {run_dir}")
            else:
                cmd_train = [
                    "yolo", "detect", "train",
                    f"model={y['base_model']}",
                    f"data={dataset_yaml}",
                    "verbose=False",
                    f"imgsz={y['imgsz']}",
                    f"epochs={y['epochs']}",
                    f"batch={y['batch']}",
                    f"patience={y['patience']}",
                    f"optimizer={y['optimizer']}",
                    f"lr0={y['lr0']}",
                    f"freeze={y.get('freeze', 0)}",
                    f"device={y.get('device', 'cpu')}",
                    f"seed={seed}",
                    f"workers={workers}",
                    "save=True",
                    "plots=True",
                    f"project={runs_dir.as_posix()}",
                    f"name={run_name}",
                    "exist_ok=True",
                ]

                for k in OPTIONAL_KEYS:
                    if k in y:
                        val = y[k]
                        if val is None:
                            cmd_train.append(f"{k}=None")
                        else:
                            cmd_train.append(f"{k}={val}")

                if "close_mosaic" not in y:
                    cmd_train.append(f"close_mosaic={close_mosaic}")

                subprocess.run(cmd_train, check=True)

            best_pt = run_dir / "weights" / "best.pt"
            if not best_pt.exists():
                best_pt = run_dir / "weights" / "last.pt"

            if not best_pt.exists():
                raise FileNotFoundError(
                    f"No encuentro best.pt ni last.pt en {(run_dir / 'weights')} "
                    f"(variant={variant}, seed={seed})"
                )

            focus_val_name = f"{run_name}_val_{focus_class_name}"
            focus_val_dir = (runs_dir / focus_val_name).resolve()
            focus_results_csv = focus_val_dir / "results.csv"

            cmd_val_focus = [
                "yolo", "detect", "val",
                f"model={best_pt.as_posix()}",
                f"data={dataset_yaml}",
                "split=val",
                f"classes={focus_class_id}",
                "save=False",
                "plots=False",
                f"project={runs_dir.as_posix()}",
                f"name={focus_val_name}",
                "exist_ok=True",
            ]

            if focus_results_csv.exists():
                print(f"[SKIP VAL] Ya existe {focus_results_csv}")
            else:
                subprocess.run(cmd_val_focus, check=True)

            runs_index.append({
                "variant": variant,
                "seed": seed,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "dataset_yaml": dataset_yaml,
                "best_pt": str(best_pt),
                "focus_class_name": focus_class_name,
                "focus_class_id": focus_class_id,
                "focus_val_dir": str(focus_val_dir),
            })

    pd.DataFrame(runs_index).to_csv(meta_dir / "runs" / "runs_index.csv", index=False)
    print(f"\n[OK] runs_index.csv guardado en {meta_dir / 'runs' / 'runs_index.csv'}")


if __name__ == "__main__":
    main()