from __future__ import annotations

import argparse
import itertools
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def write_yaml(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp4"].get("stage", "coarse")
    stages = cfg["exp4"].get("stages", {})
    if stage not in stages:
        raise ValueError(
            f"Stage '{stage}' no existe en exp4.stages. Disponibles: {list(stages.keys())}"
        )
    return stage


def variant_name(
    stage: str,
    imgsz: int,
    batch: int,
    lr0: float,
    optimizer: str,
    weight_decay: float | None = None,
    warmup_epochs: int | None = None,
) -> str:
    """
    Nombre de variante (incluye hiperparámetros clave para trazabilidad y para poder parsearlos en fine si hace falta).

    Ejemplo:
      coarse_hp_img960_b8_adamw_lr0015_wd0005_wu3
    """
    lr_tag = f"{lr0:.4f}".replace(".", "")
    opt = str(optimizer).lower()
    base = f"{stage}_hp_img{imgsz}_b{batch}_{opt}_lr{lr_tag}"

    parts = [base]

    if weight_decay is not None:
        wd_tag = f"{int(round(float(weight_decay) * 1e4)):04d}"
        parts.append(f"wd{wd_tag}")

    if warmup_epochs is not None:
        parts.append(f"wu{int(warmup_epochs)}")

    return "_".join(parts)


def expand_env_tokens(s: str) -> str:
    """
    Expande variables tipo:
      - %VAR%  (Windows CMD)
      - $VAR / ${VAR} (bash)  -> incluso en Windows, lo resolvemos manualmente
    """
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
    """
    Consideramos 'ruta' solo si contiene separadores de directorio o patrón de ruta explícita.
    Un nombre simple tipo 'yolo26s.pt' NO es ruta (para permitir descarga automática).
    """
    s = str(s).strip()

    if "/" in s or "\\" in s:
        return True

    if s.startswith(("./", ".\\")):
        return True

    if re.match(r"^[A-Za-z]:\\", s):
        return True

    return False


def ensure_model_in_project_root(root: Path, model_name: str) -> str:
    """
    Garantiza que un modelo tipo 'yolo26s.pt' exista en el project_root.
    - Si ya existe en root/model_name: lo devuelve.
    - Si no existe: intenta descargarlo vía Ultralytics y lo copia a root/model_name.
    Devuelve ruta absoluta (posix) al .pt dentro del project_root.
    """
    model_name = str(model_name).strip()
    dst = (root / model_name).resolve()

    if dst.exists():
        return dst.as_posix()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            f"No puedo importar ultralytics para descargar '{model_name}'. "
            f"Instala/activa el entorno correcto. Error: {type(e).__name__}: {e}"
        )

    try:
        y = YOLO(model_name) 

        src = None
        for attr in ("ckpt_path", "model_path", "pt_path", "path"):
            if hasattr(y, attr):
                src = getattr(y, attr)
                if src:
                    break
        if src is None and hasattr(y, "model"):
            for attr in ("ckpt_path", "model_path", "pt_path", "path"):
                if hasattr(y.model, attr):
                    src = getattr(y.model, attr)
                    if src:
                        break

        if src is None:
            print(f"[WARN] No pude localizar el path del checkpoint descargado para '{model_name}'.")
            print("       Ultralytics puede haberlo cacheado internamente. Continuaré usando el nombre.")
            return model_name

        srcp = Path(str(src))
        if not srcp.exists():
            print(f"[WARN] El checkpoint reportado por Ultralytics no existe: {srcp}")
            print("       Continuaré usando el nombre y Ultralytics lo resolverá en train.")
            return model_name

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(srcp, dst)
        print(f"[OK] Modelo descargado y guardado en project_root: {dst}")
        return dst.as_posix()

    except Exception as e:
        print(
            f"[WARN] No pude descargar/copiar '{model_name}' a project_root.\n"
            f"       Continuaré usando el nombre y Ultralytics lo resolverá en train.\n"
            f"       Motivo: {type(e).__name__}: {e}"
        )
        return model_name


def resolve_model_init(root: Path, model_raw: str) -> str:
    """
    - Expande env tokens
    - Si parece ruta, la convierte a absoluta (relativa a root si hace falta) y exige que exista.
    - Si NO parece ruta (p.ej. 'yolo26s.pt'), garantiza que esté guardado en project_root.
    """
    model_raw = expand_env_tokens(model_raw)

    if re.search(r"\$\w+|\$\{\w+\}|%\w+%", model_raw):
        raise ValueError(
            f"model contiene variables sin resolver: '{model_raw}'.\n"
            f"¿Has exportado la variable? (ej: set BASE_BEST_PT=...)\n"
        )

    if looks_like_path(model_raw):
        mp = Path(model_raw)
        if not mp.is_absolute():
            mp = (root / mp).resolve()
        model_path = mp.as_posix()
        if not Path(model_path).exists():
            raise FileNotFoundError(f"model_init no existe: {model_path}")
        return model_path

    return ensure_model_in_project_root(root, model_raw)


def resolve_dataset_yaml(root: Path, cfg: dict, dataset_variant_override: str | None) -> tuple[Path, str]:
    """
    Prioridad:
      1) --dataset_variant (CLI)
      2) exp4.dataset_variant + paths.datasets[dataset_variant]
      3) paths.base_dataset_yaml (fallback legacy)
    """
    exp4 = cfg.get("exp4", {}) or {}
    paths = cfg.get("paths", {}) or {}
    ds_map = paths.get("datasets", {}) or {}

    ds_variant = (dataset_variant_override or exp4.get("dataset_variant", "") or "").strip()
    if ds_variant:
        if ds_variant not in ds_map:
            raise KeyError(
                f"dataset_variant='{ds_variant}' no existe en paths.datasets.\n"
                f"Disponibles: {list(ds_map.keys())}"
            )
        ds_rel = Path(str(ds_map[ds_variant]).replace("\\", "/"))
        ds_path = ds_rel if ds_rel.is_absolute() else (root / ds_rel).resolve()
        if not ds_path.exists():
            raise FileNotFoundError(f"paths.datasets['{ds_variant}'] apunta a {ds_path} pero no existe.")
        return ds_path, ds_variant

    base_rel = Path(str(paths.get("base_dataset_yaml", "")).replace("\\", "/"))
    if not str(base_rel):
        raise KeyError(
            "No encuentro dataset. Define exp4.dataset_variant+paths.datasets o paths.base_dataset_yaml."
        )
    ds_path = base_rel if base_rel.is_absolute() else (root / base_rel).resolve()
    if not ds_path.exists():
        raise FileNotFoundError(f"base_dataset_yaml no existe: {ds_path}")
    return ds_path, "base_dataset_yaml"


def parse_hparams_from_variant(
    vname: str, default_imgsz: int = 960, default_batch: int = 8
) -> tuple[int, int, float, str, float, int]:
    """
    vname esperado:
      <stage>_hp_img960_b8_adamw_lr0015_wd0005_wu3

    Devuelve:
      (imgsz, batch, lr0, optimizer, weight_decay, warmup_epochs)
    """
    imgsz = default_imgsz
    batch = default_batch
    lr0 = None
    optimizer = None
    weight_decay = 0.0005
    warmup_epochs = 3

    parts = vname.split("_")
    for p in parts:
        if p.startswith("img"):
            try:
                imgsz = int(p.replace("img", ""))
            except Exception:
                pass
        if p.startswith("b"):
            try:
                batch = int(p.replace("b", ""))
            except Exception:
                pass
        if p in ("adamw", "sgd", "lion"):
            if p == "adamw":
                optimizer = "AdamW"
            elif p == "sgd":
                optimizer = "SGD"
            elif p == "lion":
                optimizer = "Lion"
        if p.startswith("lr"):
            lr_tag = p.replace("lr", "")
            try:
                lr0 = float("0." + lr_tag)
            except Exception:
                lr0 = None
        if p.startswith("wd"):
            wd_tag = p.replace("wd", "")
            try:
                weight_decay = int(wd_tag) / 1e4
            except Exception:
                pass
        if p.startswith("wu"):
            try:
                warmup_epochs = int(p.replace("wu", ""))
            except Exception:
                pass

    if optimizer is None:
        optimizer = "AdamW"
    if lr0 is None:
        lr0 = 0.0015

    return imgsz, batch, float(lr0), str(optimizer), float(weight_decay), int(warmup_epochs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine | roi_train | crop_coarse | crop_fine | hn_fine ...")
    ap.add_argument("--dataset_variant", default=None, help="Override dataset_variant (ej: v5_full, v5_acera_only, etc.)")
    ap.add_argument("--model", default=None, help="Override model init (puede ser %BASE_BEST_PT% o $BASE_BEST_PT)")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    stage_cfg = cfg["exp4"]["stages"][stage]

    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()
    runs_dir = (root / cfg["paths"]["runs_dir"] / stage).resolve()

    (meta_dir / "runs").mkdir(parents=True, exist_ok=True)
    (meta_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (meta_dir / "plots").mkdir(parents=True, exist_ok=True)

    dataset_yaml, dataset_variant = resolve_dataset_yaml(root, cfg, args.dataset_variant)

    model_raw = args.model if args.model is not None else str(cfg["exp4"]["model"])
    model_init = resolve_model_init(root, model_raw)

    freeze_default = int(cfg["exp4"]["freeze"])
    device = str(cfg.get("yolo", {}).get("device", 0))

    rows = []

    if stage == "coarse":
        grid = stage_cfg["grid"]

        imgsz_list = [int(x) for x in grid["imgsz"]]
        batch_list = [int(x) for x in grid["batch"]]
        lr0_list = [float(x) for x in grid["lr0"]]
        opt_list = [str(x) for x in grid["optimizer"]]

        wd_list = [float(x) for x in grid.get("weight_decay", [0.0005])]
        wu_list = [int(x) for x in grid.get("warmup_epochs", [3])]
        fr_list = [int(x) for x in grid.get("freeze", [freeze_default])]

        seeds = [int(s) for s in stage_cfg["seeds"]]
        epochs = int(stage_cfg["epochs"])

        combos = list(itertools.product(imgsz_list, batch_list, lr0_list, opt_list, wd_list, wu_list, fr_list))
        for imgsz, batch, lr0, optimizer, weight_decay, warmup_epochs, freeze in combos:
            vname = variant_name(stage, imgsz, batch, lr0, optimizer, weight_decay, warmup_epochs)
            for seed in seeds:
                run_name = f"exp4_{stage}_{dataset_variant}_{vname}_seed{seed}"
                run_dir = (runs_dir / run_name).resolve()
                rows.append(
                    {
                        "experiment": "exp4",
                        "stage": stage,
                        "dataset_variant": dataset_variant,
                        "variant": vname,
                        "seed": seed,
                        "run_name": run_name,
                        "run_dir": run_dir.as_posix(),
                        "dataset_yaml": dataset_yaml.as_posix(),
                        "model": model_init,
                        "freeze": int(freeze),
                        "epochs": epochs,
                        "batch": batch,
                        "imgsz": imgsz,
                        "device": device,
                        "lr0": float(lr0),
                        "optimizer": str(optimizer),
                        "weight_decay": float(weight_decay),
                        "warmup_epochs": int(warmup_epochs),
                        "da_policy": "da_baseline",
                    }
                )

    elif stage == "fine":
        topk = int(stage_cfg.get("use_topk_from_coarse", 3))
        coarse_metrics_dir = (root / cfg["paths"]["meta_dir"] / "coarse" / "metrics").resolve()
        coarse_rank = coarse_metrics_dir / "winner_focus_val_def_ranking.csv"
        if not coarse_rank.exists():
            raise FileNotFoundError(
                f"No encuentro ranking coarse en {coarse_rank}. Ejecuta coarse eval+winner primero."
            )

        df_rank = pd.read_csv(coarse_rank)
        if df_rank.empty or "variant" not in df_rank.columns:
            raise RuntimeError(f"Ranking inválido: {coarse_rank}")

        top_variants = df_rank["variant"].head(topk).tolist()

        seeds = [int(s) for s in stage_cfg["seeds"]]
        epochs = int(stage_cfg["epochs"])

        for vname in top_variants:
            sub = df_rank[df_rank["variant"] == vname].head(1)

            have_core = {"imgsz", "batch", "lr0", "optimizer"}.issubset(set(df_rank.columns))
            have_wd = "weight_decay" in df_rank.columns
            have_wu = "warmup_epochs" in df_rank.columns
            have_fr = "freeze" in df_rank.columns

            if have_core:
                imgsz = int(sub["imgsz"].iloc[0])
                batch = int(sub["batch"].iloc[0])
                lr0 = float(sub["lr0"].iloc[0])
                optimizer = str(sub["optimizer"].iloc[0])
                weight_decay = float(sub["weight_decay"].iloc[0]) if have_wd else None
                warmup_epochs = int(sub["warmup_epochs"].iloc[0]) if have_wu else None
                freeze = int(sub["freeze"].iloc[0]) if have_fr else freeze_default
            else:
                imgsz, batch, lr0, optimizer, weight_decay, warmup_epochs = parse_hparams_from_variant(vname)
                freeze = freeze_default

            if weight_decay is None:
                weight_decay = 0.0005
            if warmup_epochs is None:
                warmup_epochs = 3

            for seed in seeds:
                run_name = f"exp4_{stage}_{dataset_variant}_{vname}_seed{seed}"
                run_dir = (runs_dir / run_name).resolve()
                rows.append(
                    {
                        "experiment": "exp4",
                        "stage": stage,
                        "dataset_variant": dataset_variant,
                        "variant": vname,
                        "seed": seed,
                        "run_name": run_name,
                        "run_dir": run_dir.as_posix(),
                        "dataset_yaml": dataset_yaml.as_posix(),
                        "model": model_init,
                        "freeze": int(freeze),
                        "epochs": epochs,
                        "batch": batch,
                        "imgsz": imgsz,
                        "device": device,
                        "lr0": float(lr0),
                        "optimizer": str(optimizer),
                        "weight_decay": float(weight_decay),
                        "warmup_epochs": int(warmup_epochs),
                        "da_policy": "da_baseline",
                    }
                )

    else:
        fixed = stage_cfg.get("fixed", {}) or {}

        seeds = [int(s) for s in stage_cfg.get("seeds", [41])]
        epochs = int(stage_cfg.get("epochs", 1))
        imgsz = int(fixed.get("imgsz", 1280))
        batch = int(fixed.get("batch", 8))
        lr0 = float(fixed.get("lr0", 0.0015))
        optimizer = str(fixed.get("optimizer", "AdamW"))
        weight_decay = float(fixed.get("weight_decay", 0.0005))
        warmup_epochs = int(fixed.get("warmup_epochs", 3))
        freeze = int(fixed.get("freeze", freeze_default))

        vname = variant_name(stage, imgsz, batch, lr0, optimizer, weight_decay, warmup_epochs)
        for seed in seeds:
            run_name = f"exp4_{stage}_{dataset_variant}_{vname}_seed{seed}"
            run_dir = (runs_dir / run_name).resolve()
            rows.append(
                {
                    "experiment": "exp4",
                    "stage": stage,
                    "dataset_variant": dataset_variant,
                    "variant": vname,
                    "seed": seed,
                    "run_name": run_name,
                    "run_dir": run_dir.as_posix(),
                    "dataset_yaml": dataset_yaml.as_posix(),
                    "model": model_init,
                    "freeze": int(freeze),
                    "epochs": epochs,
                    "batch": batch,
                    "imgsz": imgsz,
                    "device": device,
                    "lr0": float(lr0),
                    "optimizer": optimizer,
                    "weight_decay": float(weight_decay),
                    "warmup_epochs": int(warmup_epochs),
                    "da_policy": "da_baseline",
                }
            )

    df = pd.DataFrame(rows)
    out_csv = meta_dir / "runs" / "runs_index.csv"
    df.to_csv(out_csv, index=False)

    ds_index = {str(v): dataset_yaml.as_posix() for v in sorted(df["variant"].unique().tolist())}
    write_yaml(meta_dir / "runs" / "datasets_index.yaml", ds_index)

    print("[OK] Exp4 preparado")
    print(f"[OK] stage={stage}")
    print(f"[OK] dataset_variant={dataset_variant}")
    print(f"[OK] dataset_yaml={dataset_yaml}")
    print(f"[OK] model_init={model_init}")
    print(f"[OK] runs_index={out_csv}")
    print(f"[OK] total runs={len(df)} | variants={df['variant'].nunique()}")
    print(df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
