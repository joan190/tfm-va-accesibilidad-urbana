from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml
from ultralytics import YOLO


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def norm(s: str) -> str:
    return str(s).strip().lower()


def find_class_id(names, target_class: str) -> int | None:
    t = norm(target_class)
    if isinstance(names, (list, tuple)):
        for i, v in enumerate(names):
            if norm(v) == t:
                return int(i)
        return None
    if isinstance(names, dict):
        for k, v in names.items():
            if norm(v) == t:
                return int(k)
        return None
    try:
        for i, v in enumerate(list(names)):
            if norm(v) == t:
                return int(i)
    except Exception:
        pass
    return None


def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp3"].get("stage", "coarse")
    stages = cfg["exp3"].get("stages", {})
    if stage not in stages:
        raise ValueError(f"Stage '{stage}' no existe en exp3.stages. Disponibles: {list(stages.keys())}")
    return stage


def conf_tag(conf: float | None) -> str:
    if conf is None:
        return "def"
    return f"conf{int(round(conf * 100)):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--target_class", default="obstaculo")
    ap.add_argument("--weights", default="best", choices=["best", "last"])
    ap.add_argument("--conf", type=float, default=None, help="Confidence threshold for val (e.g., 0.25). If omitted uses default.")
    ap.add_argument("--force", action="store_true", help="Recalcula aunque exista CSV")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()

    runs_index = meta_dir / "runs" / "runs_index.csv"
    if not runs_index.exists():
        raise FileNotFoundError(f"No encuentro {runs_index}. Ejecuta make_yolo_dataset_exp3.py antes.")

    out_dir = meta_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = conf_tag(args.conf)
    out_all = out_dir / f"metrics_{args.weights}_all_{args.split}_{tag}.csv"
    out_focus = out_dir / f"metrics_{args.weights}_focus_{args.split}_{tag}.csv"

    if out_all.exists() and out_focus.exists() and not args.force:
        print(f"[SKIP] Ya existen:\n {out_all}\n {out_focus}\nUsa --force si quieres recalcular.")
        return

    df_runs = pd.read_csv(runs_index)
    imgsz = int(cfg["yolo"]["imgsz"])
    device = str(cfg["yolo"]["device"])

    rows_all = []
    rows_focus = []

    for _, r in df_runs.iterrows():
        variant = str(r["variant"])
        seed = int(r["seed"])
        freeze = int(r.get("freeze", -1))

        run_dir = Path(str(r["run_dir"]))
        dataset_yaml = Path(str(r["dataset_yaml"]))
        wfile = run_dir / "weights" / f"{args.weights}.pt"

        if not wfile.exists():
            print(f"[WARN] No existe {args.weights}.pt: {wfile}")
            continue
        if not dataset_yaml.exists():
            print(f"[WARN] No existe dataset_yaml: {dataset_yaml}")
            continue

        print(
            f"\n[VAL Exp3] stage={stage} | {variant} seed={seed} freeze={freeze} "
            f"weights={args.weights}.pt split={args.split} conf={args.conf}"
        )

        model = YOLO(wfile.as_posix())

        val_kwargs = dict(
            data=dataset_yaml.as_posix(),
            split=args.split,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        if args.conf is not None:
            val_kwargs["conf"] = float(args.conf)

        metrics = model.val(**val_kwargs)

        # ALL
        all_dict = dict(metrics.results_dict)
        all_dict.update(
            {
                "stage": stage,
                "variant": variant,
                "seed": seed,
                "freeze": freeze,
                "run_name": r["run_name"],
                "run_dir": str(run_dir),
                "weights": args.weights,
                "split": args.split,
                "conf": args.conf if args.conf is not None else "default",
            }
        )
        rows_all.append(all_dict)

        cid = find_class_id(metrics.names, args.target_class)
        if cid is None:
            print(f"[WARN] La clase '{args.target_class}' no está en names: {metrics.names}")
            continue

        p, rcl, ap50, ap = metrics.class_result(cid)
        focus_dict = {
            "metrics/precision(B)": float(p),
            "metrics/recall(B)": float(rcl),
            "metrics/mAP50(B)": float(ap50),
            "metrics/mAP50-95(B)": float(ap),
            "stage": stage,
            "variant": variant,
            "seed": seed,
            "freeze": freeze,
            "run_name": r["run_name"],
            "run_dir": str(run_dir),
            "weights": args.weights,
            "split": args.split,
            "conf": args.conf if args.conf is not None else "default",
            "target_class": args.target_class,
            "target_class_id": int(cid),
        }
        rows_focus.append(focus_dict)

    df_all = pd.DataFrame(rows_all)
    df_focus = pd.DataFrame(rows_focus)

    if df_all.empty:
        raise RuntimeError("metrics_all vacío. ¿Se validó algo? Revisa pesos/dataset.yaml/device.")
    if df_focus.empty:
        raise RuntimeError("metrics_focus vacío. ¿Existe la clase 'obstaculo'?")

    df_all.to_csv(out_all, index=False)
    df_focus.to_csv(out_focus, index=False)

    print(f"\n[OK] Guardado ALL:   {out_all}")
    print(f"[OK] Guardado FOCUS: {out_focus}")


if __name__ == "__main__":
    main()
