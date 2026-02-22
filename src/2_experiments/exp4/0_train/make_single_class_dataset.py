from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import yaml


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def write_yaml(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def norm(s: str) -> str:
    return str(s).strip().lower()


def resolve_dataset_yaml_from_config(root: Path, cfg: dict, base_variant: str) -> Path:
    ds_map = (cfg.get("paths", {}) or {}).get("datasets", {}) or {}
    if base_variant not in ds_map:
        raise KeyError(
            f"base_dataset_variant='{base_variant}' no está en paths.datasets. Keys={list(ds_map.keys())}"
        )
    p = Path(str(ds_map[base_variant]).replace("\\", "/"))
    return p if p.is_absolute() else (root / p).resolve()


def resolve_ds_root(ds_yaml: Path, ds_cfg: dict) -> Path:
    base = Path(str(ds_cfg.get("path", ".")).replace("\\", "/"))
    return base if base.is_absolute() else (ds_yaml.parent / base).resolve()


def list_names(ds_cfg: dict) -> list[str]:
    names = ds_cfg.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names)


def hardlink_or_copy(src: Path, dst: Path, hardlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if hardlink:
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_dataset_variant", required=True, help="Ej: v4_full (key en paths.datasets)")
    ap.add_argument("--target_class", required=True, help="Ej: acera")
    ap.add_argument(
        "--out_variant",
        required=True,
        help="Ej: v4_acera_only (key en paths.datasets o se crea bajo meta/exp4/datasets)",
    )
    ap.add_argument("--hardlink", action="store_true")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    base_yaml = resolve_dataset_yaml_from_config(root, cfg, args.base_dataset_variant)
    if not base_yaml.exists():
        raise FileNotFoundError(f"Base dataset.yaml no existe: {base_yaml}")

    base_cfg = read_yaml(base_yaml)
    base_root = resolve_ds_root(base_yaml, base_cfg)
    names = list_names(base_cfg)

    t = norm(args.target_class)
    if t not in [norm(x) for x in names]:
        raise KeyError(f"target_class='{args.target_class}' no existe en names={names}")
    target_id = [norm(x) for x in names].index(t)

    out_map = (cfg.get("paths", {}) or {}).get("datasets", {}) or {}
    if args.out_variant in out_map:
        out_yaml = Path(str(out_map[args.out_variant]).replace("\\", "/"))
        out_yaml = out_yaml if out_yaml.is_absolute() else (root / out_yaml).resolve()
        out_root = out_yaml.parent
    else:
        out_root = (root / "data" / "meta" / "exp4" / "datasets" / args.out_variant).resolve()
        out_yaml = out_root / "dataset.yaml"

    for split in ["train", "val", "test"]:
        if split not in base_cfg:
            continue

        img_rel = Path(str(base_cfg[split]).replace("\\", "/"))
        img_dir = img_rel if img_rel.is_absolute() else (base_root / img_rel).resolve()

        lab_dir = (base_root / "labels" / split).resolve()
        if not lab_dir.exists():
            lab_dir = Path(str(img_dir).replace("images", "labels")).resolve()

        out_img_dir = out_root / "images" / split
        out_lab_dir = out_root / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lab_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            print(f"[AVISO] split={split}: no existe el directorio de imágenes {img_dir}, se omite")
            continue

        imgs = sorted(
            [p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        )
        for im in imgs:
            hardlink_or_copy(im, out_img_dir / im.name, args.hardlink)

            src_lab = lab_dir / f"{im.stem}.txt"
            dst_lab = out_lab_dir / f"{im.stem}.txt"

            if not src_lab.exists():
                dst_lab.write_text("", encoding="utf-8")
                continue

            out_lines = []
            for line in src_lab.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                if cid == target_id:
                    parts[0] = "0"
                    out_lines.append(" ".join(parts))

            dst_lab.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    out_cfg = {
        "path": out_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": [args.target_class],
    }
    write_yaml(out_yaml, out_cfg)

    print("[OK] Dataset de una sola clase creado")
    print(f" base: {base_yaml}")
    print(f" out:  {out_yaml}")
    print(f" clase: {args.target_class} (id={target_id} -> 0)")


if __name__ == "__main__":
    main()