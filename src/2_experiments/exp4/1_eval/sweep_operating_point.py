from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="fine", choices=["coarse", "fine"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--weights", default="best", choices=["best", "last"])
    ap.add_argument("--conf_min", type=float, default=0.05)
    ap.add_argument("--conf_max", type=float, default=0.50)
    ap.add_argument("--conf_step", type=float, default=0.05)
    ap.add_argument("--iou_list", type=str, default="0.5,0.6,0.7")
    ap.add_argument("--target_class", default="obstaculo")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    meta_dir = (root / cfg["paths"]["meta_dir"] / args.stage).resolve()
    metrics_dir = meta_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    winner_json = metrics_dir / f"winner_focus_val_def.json"
    if not winner_json.exists():
        raise FileNotFoundError(f"No encuentro {winner_json}. Ejecuta winner antes en {args.stage}.")

    import json
    winner = json.loads(winner_json.read_text(encoding="utf-8"))
    winner_variant = winner["winner_variant"]

    runs_index = meta_dir / "runs" / "runs_index.csv"
    df = pd.read_csv(runs_index)
    sub = df[df["variant"] == winner_variant].copy()
    if sub.empty:
        raise RuntimeError(f"No encuentro variant={winner_variant} en {runs_index}")

    run_dir = Path(sub.iloc[0]["run_dir"])
    dataset_yaml = Path(sub.iloc[0]["dataset_yaml"])
    wfile = run_dir / "weights" / f"{args.weights}.pt"
    if not wfile.exists():
        raise FileNotFoundError(f"No existe weights: {wfile}")

    device = str(cfg["yolo"]["device"])
    imgsz = int(sub.iloc[0]["imgsz"])

    confs = np.arange(args.conf_min, args.conf_max + 1e-9, args.conf_step).round(3).tolist()
    ious = [float(x) for x in args.iou_list.split(",")]

    model = YOLO(wfile.as_posix())

    rows = []
    for conf in confs:
        for iou in ious:
            m = model.val(
                data=dataset_yaml.as_posix(),
                split=args.split,
                imgsz=imgsz,
                device=device,
                verbose=False,
                conf=float(conf),
                iou=float(iou),
            )

            names = m.names
            cid = None
            t = args.target_class.strip().lower()
            if isinstance(names, dict):
                for k, v in names.items():
                    if str(v).strip().lower() == t:
                        cid = int(k)
                        break
            else:
                for i, v in enumerate(list(names)):
                    if str(v).strip().lower() == t:
                        cid = i
                        break
            if cid is None:
                raise RuntimeError(f"No encuentro clase '{args.target_class}' en names={names}")

            p, r, ap50, ap = m.class_result(cid)
            p = float(p); r = float(r); ap50 = float(ap50); ap = float(ap)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            rows.append(
                {
                    "variant": winner_variant,
                    "weights": args.weights,
                    "split": args.split,
                    "imgsz": imgsz,
                    "conf": conf,
                    "iou": iou,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "ap50": ap50,
                    "ap": ap,
                }
            )
            print(f"[OK] conf={conf:.2f} iou={iou:.2f} | P={p:.3f} R={r:.3f} F1={f1:.3f}")

    out_csv = metrics_dir / f"operating_sweep_{args.stage}_{args.split}_{args.weights}.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)

    best = df_out.sort_values(["f1", "recall", "precision"], ascending=False).head(1).iloc[0].to_dict()
    best_txt = metrics_dir / f"operating_best_{args.stage}_{args.split}_{args.weights}.txt"
    best_txt.write_text(str(best), encoding="utf-8")

    print(f"\n[OK] Sweep guardado: {out_csv}")
    print(f"[OK] Best (by F1): {best}")
    print(f"[OK] Best txt: {best_txt}")


if __name__ == "__main__":
    main()
