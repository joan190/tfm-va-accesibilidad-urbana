from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yaml
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def norm(s: str) -> str:
    return str(s).strip().lower()


def names_to_list(names) -> list[str]:
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, (list, tuple)):
        return list(names)
    raise ValueError(f"Formato names no soportado: {type(names)}")


def resolve_root(dataset_yaml: Path, y: dict) -> Path:
    if "path" in y and y["path"] is not None:
        p = Path(str(y["path"]).replace("\\", "/"))
        if not p.is_absolute():
            p = (dataset_yaml.parent / p).resolve()
        return p
    return dataset_yaml.parent.resolve()


def resolve_split_images(root: Path, spec: str) -> list[Path]:
    p = Path(str(spec).replace("\\", "/"))
    if not p.is_absolute():
        p = (root / p).resolve()

    if p.is_dir():
        imgs = []
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                imgs.append(f)
        return sorted(imgs)

    if p.is_file():
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out = []
        for ln in lines:
            q = Path(ln.replace("\\", "/"))
            if not q.is_absolute():
                q = (root / q).resolve()
            out.append(q)
        return out

    raise FileNotFoundError(f"Split spec no existe: {p}")


def image_to_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    lower = [p.lower() for p in parts]
    if "images" in lower:
        i = lower.index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def has_class(lbl_path: Path, class_id: int) -> bool:
    if not lbl_path.exists():
        return False
    for ln in lbl_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        c = int(float(ln.split()[0]))
        if c == class_id:
            return True
    return False


def write_yaml(p: Path, obj: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_yaml", required=True)
    ap.add_argument("--weights", required=True, help="best.pt")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--target_class", default="obstaculo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=None, help="si None, usa 960")
    ap.add_argument("--repeat", type=int, default=2, help="cuántas veces repetir hardnegs en train")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dataset_yaml = Path(args.dataset_yaml).resolve()
    y = read_yaml(dataset_yaml)
    root = resolve_root(dataset_yaml, y)
    names = names_to_list(y.get("names", {}))
    if not names:
        raise ValueError("No encuentro names en dataset.yaml")

    t = norm(args.target_class)
    cid = None
    for i, n in enumerate(names):
        if norm(n) == t:
            cid = i
            break
    if cid is None:
        raise ValueError(f"No encuentro clase '{args.target_class}' en names={names}")

    split_spec = y.get(args.split)
    if not split_spec:
        raise ValueError(f"dataset.yaml no tiene split '{args.split}'")
    imgs = resolve_split_images(root, split_spec)
    if not imgs:
        raise RuntimeError(f"No encontré imágenes en split {args.split}")

    # Nos quedamos solo con imágenes SIN obstáculo GT (negativos)
    neg_imgs = []
    for img in imgs:
        lbl = image_to_label_path(img)
        if not has_class(lbl, cid):
            neg_imgs.append(img)

    print(f"[INFO] split={args.split} total_imgs={len(imgs)} | negatives(no obstaculo)={len(neg_imgs)}")

    model = YOLO(str(Path(args.weights).resolve()))
    imgsz = args.imgsz if args.imgsz is not None else 960

    rows = []
    hardneg_imgs = []

    for img in neg_imgs:
        res = model.predict(
            source=img.as_posix(),
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(imgsz),
            device=str(args.device),
            verbose=False,
        )
        r = res[0]
        max_conf = 0.0
        num = 0
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            mask = (cls == cid)
            if mask.any():
                num = int(mask.sum())
                max_conf = float(confs[mask].max())
        if num > 0:
            hardneg_imgs.append(img)
            rows.append({"image": img.as_posix(), "num_fp": num, "max_conf": max_conf})

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values(["max_conf", "num_fp"], ascending=False)
    csv_path = out_dir / f"hardneg_{args.split}_conf{int(args.conf*100):02d}.csv"
    df.to_csv(csv_path, index=False)

    print(f"[OK] HardNeg candidates: {len(df)}")
    print(f"[OK] CSV: {csv_path}")

    def read_split_list(spec: str) -> list[str]:
        p = Path(str(spec).replace("\\", "/"))
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.is_dir():
            imgs2 = []
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    imgs2.append(f.as_posix())
            return sorted(imgs2)
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out = []
        for ln in lines:
            q = Path(ln.replace("\\", "/"))
            if not q.is_absolute():
                q = (root / q).resolve()
            out.append(q.as_posix())
        return out

    base_train = read_split_list(y["train"])
    base_val = read_split_list(y.get("val", y["train"]))
    base_test = read_split_list(y.get("test", y.get("val", y["train"])))

    hard_list = [p.as_posix() for p in hardneg_imgs]
    train_new = base_train + hard_list * int(args.repeat)

    train_txt = out_dir / "train_hardneg.txt"
    val_txt = out_dir / "val.txt"
    test_txt = out_dir / "test.txt"
    train_txt.write_text("\n".join(train_new) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(base_val) + "\n", encoding="utf-8")
    test_txt.write_text("\n".join(base_test) + "\n", encoding="utf-8")

    out_yaml = out_dir / "dataset_hardneg.yaml"
    out_obj = {
        "path": "",
        "train": train_txt.as_posix(),
        "val": val_txt.as_posix(),
        "test": test_txt.as_posix(),
        "names": y.get("names"),
    }
    write_yaml(out_yaml, out_obj)

    print(f"[OK] dataset_hardneg.yaml: {out_yaml}")
    print(f"[INFO] oversampling repeat={args.repeat} | added={len(hard_list)} unique hardnegs")


if __name__ == "__main__":
    main()
