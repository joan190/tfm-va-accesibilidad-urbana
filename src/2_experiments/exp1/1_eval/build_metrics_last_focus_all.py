from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def resolve_path(root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def load_names_from_data_yaml(data_yaml: Path) -> list[str]:
    d = read_yaml(data_yaml)
    names = d.get("names", None)
    if names is None:
        raise ValueError(f"El YAML {data_yaml} no tiene 'names'.")

    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
        return [v for _, v in items]

    if isinstance(names, list):
        return names

    raise TypeError(f"Formato 'names' no soportado en {data_yaml}: {type(names)}")


def safe_getattr(obj, candidates: list[str]):
    for c in candidates:
        if hasattr(obj, c):
            return getattr(obj, c)
    return None


def extract_all_metrics(val_results) -> dict:
    box = getattr(val_results, "box", None)
    if box is None:
        raise ValueError("No encuentro val_results.box. ¿Ha cambiado la versión de ultralytics?")

    mp = safe_getattr(box, ["mp", "mean_precision"])
    mr = safe_getattr(box, ["mr", "mean_recall"])
    map50 = safe_getattr(box, ["map50"])
    map_5095 = safe_getattr(box, ["map"])

    if mp is None or mr is None or map50 is None or map_5095 is None:
        raise ValueError(
            "No puedo extraer las métricas ALL desde val_results.box. "
            "Revisa los atributos disponibles en tu versión de ultralytics."
        )

    return {
        "metrics/precision": float(mp),
        "metrics/recall": float(mr),
        "metrics/mAP50": float(map50),
        "metrics/mAP50-95": float(map_5095),
    }


def extract_focus_metrics(val_results, class_idx: int) -> dict:
    box = getattr(val_results, "box", None)
    if box is None:
        raise ValueError("No encuentro val_results.box.")

    p = safe_getattr(box, ["p"])
    r = safe_getattr(box, ["r"])
    ap50 = safe_getattr(box, ["ap50"])
    ap = safe_getattr(box, ["ap", "maps"])

    if p is None or r is None or ap50 is None or ap is None:
        raise ValueError(
            "No puedo extraer métricas FOCUS por clase (p/r/ap50/ap). "
            "Puede variar según la versión de ultralytics."
        )

    return {
        "metrics/precision": float(p[class_idx]),
        "metrics/recall": float(r[class_idx]),
        "metrics/mAP50": float(ap50[class_idx]),
        "metrics/mAP50-95": float(ap[class_idx]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--target_class", default="obstaculo")
    ap.add_argument("--weights", default="best", choices=["best", "last"])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]

    runs_index = pd.read_csv(meta_dir / "runs" / "runs_index.csv")

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "No puedo importar ultralytics. Instálalo con:\n"
            "  pip install ultralytics\n"
            f"Error: {e}"
        )

    rows_all = []
    rows_focus = []

    for _, r in runs_index.iterrows():
        run_dir = resolve_path(root, r["run_dir"])
        weights_path = run_dir / "weights" / f"{args.weights}.pt"

        if not weights_path.exists():
            print(f"[WARN] No existe {weights_path}. Se omite el run: {run_dir}")
            continue

        data_yaml = None
        args_yaml = run_dir / "args.yaml"
        if args_yaml.exists():
            train_args = read_yaml(args_yaml)
            data_yaml = train_args.get("data", None)

        if data_yaml is None:
            print(f"[WARN] No puedo encontrar el data yaml en {args_yaml}. Se omite el run: {run_dir}")
            continue

        data_yaml_path = resolve_path(root, data_yaml)
        if not data_yaml_path.exists():
            print(f"[WARN] El data yaml no existe: {data_yaml_path}. Se omite el run: {run_dir}")
            continue

        names = load_names_from_data_yaml(data_yaml_path)
        if args.target_class not in names:
            print(f"[WARN] target_class='{args.target_class}' no está en names={names}. Se omite el run: {run_dir}")
            continue

        class_idx = names.index(args.target_class)

        model = YOLO(str(weights_path))
        val_results = model.val(
            data=str(data_yaml_path),
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            plots=False,
            save=False,
            verbose=False,
        )

        m_all = extract_all_metrics(val_results)
        row_all = {
            "variant": r.get("variant", None),
            "seed": r.get("seed", None),
            "run_name": r.get("run_name", None),
            "run_dir": str(run_dir),
            "weights": args.weights,
            "split": args.split,
            **m_all,
        }
        rows_all.append(row_all)

        m_focus = extract_focus_metrics(val_results, class_idx=class_idx)
        row_focus = {
            "variant": r.get("variant", None),
            "seed": r.get("seed", None),
            "run_name": r.get("run_name", None),
            "run_dir": str(run_dir),
            "weights": args.weights,
            "split": args.split,
            "focus_class": args.target_class,
            "focus_class_idx": class_idx,
            **m_focus,
        }
        rows_focus.append(row_focus)

        print(f"[OK] {r.get('run_name')} -> métricas ALL y FOCUS calculadas")

    df_all = pd.DataFrame(rows_all)
    df_focus = pd.DataFrame(rows_focus)

    out_all = meta_dir / "metrics" / "metrics_last_all.csv"
    out_focus = meta_dir / "metrics" / "metrics_last_focus.csv"

    df_all.to_csv(out_all, index=False)
    df_focus.to_csv(out_focus, index=False)

    print(f"\n[OK] Generado: {out_all}")
    print(f"[OK] Generado: {out_focus}\n")


if __name__ == "__main__":
    main()