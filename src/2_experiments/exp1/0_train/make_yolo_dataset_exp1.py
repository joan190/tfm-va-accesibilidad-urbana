from __future__ import annotations

import os
import argparse
from pathlib import Path
import shutil
import subprocess
import yaml
import pandas as pd

def _remove_path(p: Path) -> None:
    if not p.exists() and not p.is_symlink():
        return
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
    except FileNotFoundError:
        return


def link_dir(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()

    if not src.exists():
        raise FileNotFoundError(f"El directorio de origen no existe: {src}")

    # Asegura que exista el padre del destino y borra el destino si ya estaba
    dst.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(dst)

    if os.name != "nt":
        # Unix-like: enlace simbólico
        dst.symlink_to(src, target_is_directory=True)
        return

    # Windows: junction
    cmd = ["cmd", "/c", "mklink", "/J", str(dst), str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "No se pudo crear la junction en Windows.\n"
            f"Comando: {' '.join(cmd)}\n"
            f"STDOUT: {r.stdout}\n"
            f"STDERR: {r.stderr}\n"
            "Tip: prueba a ejecutar la terminal como Administrador si hace falta."
        )


def link_file(src: Path, dst: Path) -> None:
    """
    Crea un enlace al archivo SIN copiar.
    - Unix: symlink.
    - Windows: intenta symlink y, si falla, hardlink (recomendado, no suele requerir admin).
    """
    src = src.resolve()
    dst = dst.resolve()

    if not src.exists():
        raise FileNotFoundError(f"El archivo de origen no existe: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(dst)

    if os.name != "nt":
        dst.symlink_to(src)
        return

    # Windows: primero intenta symlink
    try:
        dst.symlink_to(src)
        return
    except OSError:
        pass

    # fallback: hardlink
    try:
        os.link(src, dst)
        return
    except OSError as e:
        raise RuntimeError(
            f"No se pudo enlazar el archivo en Windows: {src} -> {dst}. Error: {e}\n"
            "Tip: los hardlinks requieren estar en el mismo disco/volumen."
        ) from e


def link_images_files(src_dir: Path, dst_dir: Path) -> None:
    """
    Enlaza (hardlink/symlink) cada imagen de src_dir dentro de dst_dir.
    Esto evita junctions de directorio y previene problemas de resolución de rutas con Ultralytics.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".webp"}

    for p in src_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            link_file(p, dst_dir / p.name)


# ----------------------------
# Helpers de etiquetas YOLO
# ----------------------------

def xyxy_to_yolo(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0.0, min(float(xmin), float(w)))
    xmax = max(0.0, min(float(xmax), float(w)))
    ymin = max(0.0, min(float(ymin), float(h)))
    ymax = max(0.0, min(float(ymax), float(h)))

    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return (cx / w, cy / h, bw / w, bh / h)


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )


def image_ids_in_split(images_dir: Path, split: str) -> set[str]:
    d = images_dir / split
    if not d.exists():
        raise FileNotFoundError(f"No existe el directorio de imágenes para el split: {d}")
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    out = set()
    for e in exts:
        out |= {p.name for p in d.glob(e)}
    return out


def build_labels_for_variant(
    df: pd.DataFrame,
    variant: str,
    classes: list[str],
    images_dir: Path,
    labels_out_dir: Path
):
    """
    Escribe las etiquetas YOLO como TXT bajo:
      labels_out_dir/<variant>/{train,val,test}/<image_stem>.txt
    """
    class_to_id = {c: i for i, c in enumerate(classes)}
    dfv = df[df["class_name"].isin(classes)].copy()

    # Reinicia la salida de este variant para evitar archivos antiguos
    variant_root = labels_out_dir / variant
    if variant_root.exists():
        shutil.rmtree(variant_root)

    # Mapea nombre de imagen -> split según en qué carpeta (train/val/test) está
    split_map: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        for img in image_ids_in_split(images_dir, split):
            split_map[img] = split

    # Asegura directorios de salida
    for split in ["train", "val", "test"]:
        (labels_out_dir / variant / split).mkdir(parents=True, exist_ok=True)

    written = 0

    # Agrupa por image_id (robusto: si image_id trae rutas, usamos basename)
    for image_id, g in dfv.groupby("image_id", sort=False):
        img_name = Path(str(image_id)).name
        split = split_map.get(img_name)
        if split is None:
            continue

        w = float(g["image_width"].iloc[0])
        h = float(g["image_height"].iloc[0])

        lines = []
        for _, r in g.iterrows():
            cls_id = class_to_id[r["class_name"]]
            cx, cy, bw, bh = xyxy_to_yolo(
                r["xmin"], r["ymin"], r["xmax"], r["ymax"], w, h
            )
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue

        out = labels_out_dir / variant / split / f"{Path(img_name).stem}.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1

    print(f"[Exp1] Etiquetas escritas para {variant}: {written} archivos")


def make_dataset_view(
    meta_dir: Path,
    images_dir: Path,
    labels_out_dir: Path,
    variant: str,
    classes: list[str]
) -> Path:
    """
    Crea una "vista" estándar de dataset YOLO:
      meta_dir/datasets/<variant>/
        images/{train,val,test} -> ENLACES A ARCHIVOS (hardlink/symlink) a data/images/<split>/*
        labels/{train,val,test} -> ENLACE A DIRECTORIO a data/labels_yolo/exp1/<variant>/<split>
        dataset.yaml

    IMPORTANTE:
      - Las imágenes NUNCA se copian (los hardlinks no duplican datos).
      - Usar enlaces a archivos evita que Ultralytics resuelva rutas fuera de la vista (problema de junctions en Windows).
    """
    root = meta_dir / "datasets" / variant
    if root.exists():
        shutil.rmtree(root)

    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        # imágenes: enlaza archivos (NO junction de directorio)
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        link_images_files(images_dir / split, root / "images" / split)

        # labels: el enlace a directorio aquí va bien
        link_dir(labels_out_dir / variant / split, root / "labels" / split)

    dataset_yaml = {
        "path": str(root.resolve()),  # ruta absoluta
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(classes)},
    }
    write_yaml(root / "dataset.yaml", dataset_yaml)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ruta a configs/base.yaml (o configs/exp1/base.yaml)")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    annotations_csv = root / cfg["paths"]["annotations_csv"]
    images_dir = root / cfg["paths"]["images_dir"]
    labels_out_dir = root / cfg["paths"]["labels_out_dir"]
    meta_dir = root / cfg["paths"]["meta_dir"]
    variants_file = root / cfg["variants_file"]

    variants = read_yaml(variants_file)["variants"]

    df = pd.read_csv(annotations_csv)
    needed = {"image_id", "image_width", "image_height", "class_name", "xmin", "ymin", "xmax", "ymax"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en annotations.csv: {missing}")

    # 1) Generar labels (archivos reales)
    for vname, vcfg in variants.items():
        classes = vcfg["classes"]
        print(f"[Exp1] Generando etiquetas YOLO: {vname} clases={classes}")
        build_labels_for_variant(df, vname, classes, images_dir, labels_out_dir)

    # 2) Crear vistas del dataset (solo enlaces) + índice
    datasets_index = {}
    for vname, vcfg in variants.items():
        classes = vcfg["classes"]
        view_root = make_dataset_view(meta_dir, images_dir, labels_out_dir, vname, classes)
        datasets_index[vname] = str((view_root / "dataset.yaml").resolve())
        print(f"[Exp1] Vista del dataset lista: {vname} -> {datasets_index[vname]}")

    write_yaml(meta_dir / "datasets_index.yaml", datasets_index)
    print(f"[OK] datasets_index.yaml guardado en {meta_dir / 'datasets_index.yaml'}")


if __name__ == "__main__":
    main()