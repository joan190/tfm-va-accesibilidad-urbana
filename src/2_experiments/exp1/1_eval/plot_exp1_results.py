from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")  # evita ventanas/GUI si alguien deja un plt.show()
import matplotlib.pyplot as plt

from cycler import cycler


METRICS_KEEP_EN = {
    "recall", "precision", "mAP50", "mAP50-95", "mAP50_95",
    "box_loss", "cls_loss", "dfl_loss", "loss"
}

def metric_label_keep_en(s: str) -> str:
    s = str(s)
    s = s.replace("metrics/", "").replace("(B)", "")
    s = s.replace("val/", "val ").replace("train/", "train ")
    return s

def es_title_training(label_en: str) -> str:
    return f"Curvas de entrenamiento — {label_en} (media±desv. típ. entre semillas)"

def es_title_exp1(source: str, metric_en: str) -> str:
    return f"Exp1 ({source}) — {metric_en} (media±desv. típ.)"

def es_title_winner(source: str) -> str:
    return f"Exp1 ({source}) — Ranking de winner score"

def es_title_tradeoff(source: str, x_en: str, y_en: str) -> str:
    return f"Exp1 ({source}) — Compromiso ({x_en} vs {y_en})"


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_color_palette(n: int) -> list:
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n)]
    cmap = plt.get_cmap("hsv", n)
    return [cmap(i) for i in range(n)]

def setup_color_academic_style():
    colors = get_color_palette(10)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "font.family": "serif",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.linewidth": 0.6,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "legend.frameon": False,
        "axes.prop_cycle": cycler(color=colors),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

def _with_color_suffix(out_base: Path) -> Path:
    if out_base.name.endswith("_color"):
        return out_base
    return out_base.with_name(out_base.name + "_color")

def save_fig(fig, out_base: Path, tight_rect=None):
    out_base = _with_color_suffix(out_base)

    if tight_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_rect)

    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

def order_variants(df: pd.DataFrame, ranking_csv: Optional[Path]) -> list[str]:
    if ranking_csv and ranking_csv.exists():
        r = pd.read_csv(ranking_csv)
        if "variant" in r.columns:
            return list(r["variant"].astype(str))
    return sorted(df["variant"].astype(str).unique().tolist())

def linestyle_cycle(n: int) -> list:
    styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]
    return [styles[i % len(styles)] for i in range(n)]

def marker_cycle(n: int) -> list[str]:
    markers = ["o", "s", "^", "D", "v", "P", "X", "h", ">", "<"]
    return [markers[i % len(markers)] for i in range(n)]

def pick_first_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def resolve_rel(root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def legend_below(ax, ncol: int = 2):
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol,
        borderaxespad=0.0,
        handlelength=2.8,
        columnspacing=1.2,
    )

def legend_right(ax):
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        borderaxespad=0.0,
        handlelength=2.2,
    )


def plot_metric_bars(
    df_summary: pd.DataFrame,
    metric: str,
    variants_order: list[str],
    winner_variant: Optional[str],
    out_base: Path,
    title: str,
):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df_summary.columns or std_col not in df_summary.columns:
        print(f"[WARN] No encuentro {mean_col}/{std_col}. No genero el plot de {metric}")
        return

    d = df_summary.set_index("variant").reindex(variants_order).reset_index()
    x = range(len(d))
    means = d[mean_col].astype(float).values
    stds = d[std_col].astype(float).values

    colors = get_color_palette(len(d))

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bars = ax.bar(
        x, means, yerr=stds, capsize=2.5,
        color=colors,
        edgecolor="black", linewidth=0.9,
    )

    for i, b in enumerate(bars):
        if winner_variant is not None and str(d.loc[i, "variant"]) == str(winner_variant):
            b.set_linewidth(1.8)

    ax.set_title(title)
    ax.set_ylabel(metric_label_keep_en(metric))
    ax.set_xticks(list(x))
    ax.set_xticklabels(d["variant"].astype(str).tolist(), rotation=20, ha="right")
    ax.grid(True, axis="y")

    save_fig(fig, out_base)

def plot_winner_score(
    ranking_csv: Path,
    winner_variant: Optional[str],
    out_base: Path,
    title: str,
):
    if not ranking_csv.exists():
        print(f"[WARN] No existe el ranking: {ranking_csv}. No genero el plot de winner score.")
        return

    df = pd.read_csv(ranking_csv)
    if "variant" not in df.columns or "winner_score" not in df.columns:
        print("[WARN] El ranking no tiene las columnas esperadas. No genero el plot.")
        return

    variants = df["variant"].astype(str).tolist()
    scores = df["winner_score"].astype(float).values
    colors = get_color_palette(len(variants))

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bars = ax.bar(
        range(len(variants)), scores,
        color=colors,
        edgecolor="black", linewidth=0.9,
    )
    for i, b in enumerate(bars):
        if winner_variant is not None and variants[i] == str(winner_variant):
            b.set_linewidth(1.8)

    ax.set_title(title)
    ax.set_ylabel("winner_score")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.grid(True, axis="y")

    save_fig(fig, out_base)

def plot_pareto_scatter_means(
    df_summary: pd.DataFrame,
    variants_order: list[str],
    out_base: Path,
    title: str,
    x_metric: str,
    y_metric: str,
    winner_variant: Optional[str] = None,
):
    x_col = f"{x_metric}_mean"
    y_col = f"{y_metric}_mean"
    if x_col not in df_summary.columns or y_col not in df_summary.columns:
        print(f"[WARN] No encuentro {x_col}/{y_col} en el summary. No genero el scatter.")
        return

    d = df_summary.set_index("variant").reindex(variants_order).reset_index()
    d["x"] = d[x_col].astype(float)
    d["y"] = d[y_col].astype(float)
    d = d.dropna(subset=["x", "y"]).copy()
    if d.empty:
        print("[WARN] Scatter vacío tras dropna. No genero nada.")
        return

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    colors = get_color_palette(len(d))

    for i, row in enumerate(d.to_dict("records")):
        v = str(row["variant"])
        x = float(row["x"])
        y = float(row["y"])
        is_winner = (winner_variant is not None and v == str(winner_variant))

        ax.scatter(
            [x], [y],
            s=78 if is_winner else 56,
            marker="o",
            c=[colors[i]],
            edgecolors="black",
            linewidths=1.7 if is_winner else 1.0,
            zorder=3,
            label=v,
        )

    xmin, xmax = float(d["x"].min()), float(d["x"].max())
    ymin, ymax = float(d["y"].min()), float(d["y"].max())
    xpad = (xmax - xmin) * 0.07 if xmax > xmin else 0.01
    ypad = (ymax - ymin) * 0.07 if ymax > ymin else 0.01
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.set_title(title)
    ax.set_xlabel(metric_label_keep_en(x_metric))
    ax.set_ylabel(metric_label_keep_en(y_metric))
    ax.grid(True)

    legend_right(ax)
    save_fig(fig, out_base, tight_rect=[0.0, 0.0, 0.78, 1.0])

def plot_training_curve_mean_std(
    df_long: pd.DataFrame,
    variants_order: list[str],
    metric_col: str,
    out_base: Path,
    title: str,
    ylabel: str,
):
    if metric_col not in df_long.columns or "epoch" not in df_long.columns or "variant" not in df_long.columns:
        print(f"[WARN] No encuentro columnas para la curva: epoch/variant/{metric_col}. No genero nada.")
        return

    g = (
        df_long.groupby(["variant", "epoch"])[metric_col]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    colors = get_color_palette(len(variants_order))

    for i, v in enumerate(variants_order):
        sub = g[g["variant"].astype(str) == str(v)].sort_values("epoch")
        if sub.empty:
            continue
        x = sub["epoch"].astype(int).values
        y = sub["mean"].astype(float).values
        s = sub["std"].astype(float).fillna(0.0).values

        ax.plot(
            x, y,
            linestyle="-",
            linewidth=1.25,
            color=colors[i],
            label=v
        )
        ax.fill_between(
            x, y - s, y + s,
            color=colors[i],
            alpha=0.18,
            linewidth=0
        )

    ax.set_title(title)
    ax.set_xlabel("época")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    legend_below(ax, ncol=2)
    save_fig(fig, out_base)


def list_val_images_from_dataset_yaml(dataset_yaml: Path, project_root: Path) -> list[Path]:
    d = read_yaml(dataset_yaml)
    base = d.get("path", None)
    if base is None:
        base_path = dataset_yaml.parent
    else:
        base_path = resolve_rel(project_root, base)

    val_entry = d.get("val", None)
    if val_entry is None:
        raise ValueError(f"{dataset_yaml} no tiene clave 'val'.")

    val_path = (base_path / val_entry).resolve() if not Path(val_entry).is_absolute() else Path(val_entry)
    if val_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imgs = [p for p in val_path.rglob("*") if p.suffix.lower() in exts]
        return sorted(imgs)

    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in val_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out = []
        for ln in lines:
            p = Path(ln)
            if not p.is_absolute():
                p = (base_path / p).resolve()
            out.append(p)
        return [p for p in out if p.exists()]

    if val_path.is_file():
        return [val_path]

    raise FileNotFoundError(f"No puedo resolver imágenes de val desde: {val_path}")

def compute_score(row: pd.Series, weights: dict[str, float]) -> float:
    s = 0.0
    for k, w in weights.items():
        try:
            s += float(row.get(k, 0.0)) * float(w)
        except Exception:
            s += 0.0
    return float(s)

def pick_best_run_per_variant(df_runs: pd.DataFrame, variant: str, weights: dict[str, float]) -> Optional[pd.Series]:
    sub = df_runs[df_runs["variant"].astype(str) == str(variant)].copy()
    if sub.empty:
        return None
    sub["__score"] = sub.apply(lambda r: compute_score(r, weights), axis=1)
    return sub.sort_values("__score", ascending=False).iloc[0]

def render_qualitative_grid(
    runs_index: pd.DataFrame,
    df_focus_runs: pd.DataFrame,
    variants_order: list[str],
    dataset_yaml_for_sampling: Path,
    project_root: Path,
    out_dir: Path,
    n_images: int,
    conf: float,
    iou: float,
):
    if n_images <= 0:
        return

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[WARN] No puedo importar ultralytics para cualitativas. Se omite. Error: {e}")
        return

    imgs = list_val_images_from_dataset_yaml(dataset_yaml_for_sampling, project_root)
    if not imgs:
        print("[WARN] No encuentro imágenes de val para cualitativas. Se omite.")
        return

    random.seed(42)
    sampled = random.sample(imgs, k=min(n_images, len(imgs)))

    weights = {"metrics/recall": 0.6, "metrics/mAP50": 0.4}

    v2w = {}
    for v in variants_order:
        best_row = pick_best_run_per_variant(df_focus_runs, v, weights)
        if best_row is None:
            continue
        run_dir = Path(str(best_row["run_dir"]))
        w = run_dir / "weights" / "best.pt"
        if not w.exists():
            w = run_dir / "weights" / "last.pt"
        if w.exists():
            v2w[v] = w

    if not v2w:
        print("[WARN] No pude resolver pesos por variante para cualitativas. Se omite.")
        return

    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_ok = True
    except Exception:
        pil_ok = False

    font = None
    if pil_ok:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    for idx, img_path in enumerate(sampled, start=1):
        panels = []
        for v in variants_order:
            if v not in v2w:
                continue
            model = YOLO(str(v2w[v]))
            res = model.predict(source=str(img_path), conf=conf, iou=iou, save=False, verbose=False)
            if not res:
                continue
            arr = res[0].plot()
            arr = arr[..., ::-1]

            if pil_ok:
                im = Image.fromarray(arr)
                w0, h0 = im.size
                band_h = max(22, int(h0 * 0.06))
                canvas = Image.new("RGB", (w0, h0 + band_h), color=(255, 255, 255))
                canvas.paste(im, (0, band_h))
                draw = ImageDraw.Draw(canvas)
                draw.text((6, 4), v, fill=(0, 0, 0), font=font)
                panels.append(canvas)
            else:
                print("[WARN] PIL no disponible, no puedo generar grids cualitativas.")
                return

        if len(panels) < 2:
            continue

        widths = [p.size[0] for p in panels]
        heights = [p.size[1] for p in panels]
        H = max(heights)
        W = sum(widths)

        grid = Image.new("RGB", (W, H), color=(255, 255, 255))
        x0 = 0
        for p in panels:
            grid.paste(p, (x0, 0))
            x0 += p.size[0]

        out_path = out_dir / f"exp1_qual_compare_{idx:02d}_color.png"
        grid.save(out_path)

    print(f"[OK] Cualitativas guardadas en: {out_dir}")


def add_violinplot(ax, data, positions=None, widths=0.8):
    vp = ax.violinplot(
        data,
        positions=positions,
        widths=widths,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    colors = get_color_palette(len(vp["bodies"]))
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(colors[i])
        b.set_edgecolor("black")
        b.set_linewidth(0.8)
        b.set_alpha(0.85)
    if "cmedians" in vp:
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.0)
    return vp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default="focus", choices=["focus", "all"])
    ap.add_argument("--x_metric", default="metrics/mAP50")
    ap.add_argument("--y_metric", default="metrics/recall")
    ap.add_argument("--qual_n", type=int, default=0, help="N imágenes para comparación cualitativa (0=off).")
    ap.add_argument("--qual_conf", type=float, default=0.25)
    ap.add_argument("--qual_iou", type=float, default=0.6)
    args = ap.parse_args()

    setup_color_academic_style()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_root = root / cfg["paths"]["meta_dir"]
    metrics_dir = ensure_dir(meta_root / "metrics")
    plots_dir = ensure_dir(meta_root / "plots")

    summary_file = find_first_existing([
        metrics_dir / ("metrics_summary_focus.csv" if args.source == "focus" else "metrics_summary_all.csv"),
        meta_root / ("metrics_summary_focus.csv" if args.source == "focus" else "metrics_summary_all.csv"),
    ])
    if summary_file is None:
        raise FileNotFoundError("No encuentro metrics_summary_*.csv. Ejecuta antes aggregate_exp1_results.py")

    ranking_csv = find_first_existing([
        metrics_dir / f"winner_{args.source}_ranking.csv",
        meta_root / f"winner_{args.source}_ranking.csv",
    ])
    winner_json = find_first_existing([
        metrics_dir / f"winner_{args.source}.json",
        meta_root / f"winner_{args.source}.json",
    ])

    winner_variant = None
    if winner_json and winner_json.exists():
        try:
            winner_payload = read_json(winner_json)
            winner_variant = winner_payload.get("winner_variant", None)
        except Exception:
            winner_variant = None

    runs_csv = find_first_existing([
        metrics_dir / ("metrics_last_focus.csv" if args.source == "focus" else "metrics_last_all.csv"),
        meta_root / ("metrics_last_focus.csv" if args.source == "focus" else "metrics_last_all.csv"),
    ])
    long_csv = find_first_existing([
        metrics_dir / "metrics_long.csv",
        meta_root / "metrics_long.csv",
    ])
    runs_index_csv = find_first_existing([
        meta_root / "runs" / "runs_index.csv",
        metrics_dir / "runs_index.csv",
    ])

    df_summary = pd.read_csv(summary_file)
    if "variant" not in df_summary.columns:
        raise ValueError(f"{summary_file} no tiene columna 'variant'")

    variants_order = order_variants(df_summary, ranking_csv)

    key_metrics = ["metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95"]
    for m in key_metrics:
        out_base = plots_dir / f"exp1_{args.source}_bars_{m.replace('metrics/','').replace('-','_')}"
        plot_metric_bars(
            df_summary=df_summary,
            metric=m,
            variants_order=variants_order,
            winner_variant=winner_variant,
            out_base=out_base,
            title=es_title_exp1(args.source, metric_label_keep_en(m)),
        )

    if ranking_csv:
        plot_winner_score(
            ranking_csv=ranking_csv,
            winner_variant=winner_variant,
            out_base=plots_dir / f"exp1_{args.source}_winner_score_ranking",
            title=es_title_winner(args.source),
        )

    plot_pareto_scatter_means(
        df_summary=df_summary,
        variants_order=variants_order,
        out_base=plots_dir / f"exp1_{args.source}_pareto_means_custom",
        title=es_title_tradeoff(
            args.source,
            metric_label_keep_en(args.x_metric),
            metric_label_keep_en(args.y_metric),
        ),
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        winner_variant=winner_variant,
    )

    pareto_pairs = [
        ("metrics/recall",      "metrics/precision"),
        ("metrics/precision",   "metrics/recall"),
        ("metrics/mAP50",       "metrics/recall"),
        ("metrics/mAP50",       "metrics/precision"),
        ("metrics/mAP50-95",    "metrics/recall"),
        ("metrics/mAP50-95",    "metrics/precision"),
        ("metrics/mAP50",       "metrics/mAP50-95"),
    ]
    for x_m, y_m in pareto_pairs:
        x_col = f"{x_m}_mean"
        y_col = f"{y_m}_mean"
        if x_col not in df_summary.columns or y_col not in df_summary.columns:
            print(f"[WARN] No genero scatter {x_m} vs {y_m}: faltan {x_col}/{y_col} en summary.")
            continue

        out_name = f"exp1_{args.source}_pareto_means_{metric_label_keep_en(x_m).replace('-','_')}_vs_{metric_label_keep_en(y_m).replace('-','_')}"
        plot_pareto_scatter_means(
            df_summary=df_summary,
            variants_order=variants_order,
            out_base=plots_dir / out_name,
            title=es_title_tradeoff(args.source, metric_label_keep_en(x_m), metric_label_keep_en(y_m)),
            x_metric=x_m,
            y_metric=y_m,
            winner_variant=winner_variant,
        )

    if long_csv and long_csv.exists():
        df_long = pd.read_csv(long_csv)

        metric_specs = [
            (["metrics/mAP50(B)", "metrics/mAP50"], "mAP50"),
            (["metrics/recall(B)", "metrics/recall"], "recall"),
            (["metrics/precision(B)", "metrics/precision"], "precision"),
            (["metrics/mAP50-95(B)", "metrics/mAP50-95"], "mAP50-95"),
        ]
        for candidates, label in metric_specs:
            col = pick_first_col(df_long, candidates)
            if col:
                plot_training_curve_mean_std(
                    df_long=df_long,
                    variants_order=variants_order,
                    metric_col=col,
                    out_base=plots_dir / f"exp1_curve_{col.replace('/','_').replace('(','').replace(')','').replace('-','_')}",
                    title=es_title_training(label),
                    ylabel=label,
                )

        loss_specs = [
            (["train/box_loss"], "pérdida train box_loss"),
            (["train/cls_loss"], "pérdida train cls_loss"),
            (["train/dfl_loss"], "pérdida train dfl_loss"),
            (["val/box_loss"], "pérdida val box_loss"),
            (["val/cls_loss"], "pérdida val cls_loss"),
            (["val/dfl_loss"], "pérdida val dfl_loss"),
        ]
        for candidates, label in loss_specs:
            col = pick_first_col(df_long, candidates)
            if col:
                plot_training_curve_mean_std(
                    df_long=df_long,
                    variants_order=variants_order,
                    metric_col=col,
                    out_base=plots_dir / f"exp1_curve_{col.replace('/','_').replace('(','').replace(')','').replace('-','_')}",
                    title=f"Curvas de entrenamiento — {metric_label_keep_en(col)} (media±desv. típ. entre semillas)",
                    ylabel=label,
                )
    else:
        print("[WARN] No encuentro metrics_long.csv, no genero curvas por época.")

    if args.qual_n > 0:
        if runs_index_csv is None or not runs_index_csv.exists():
            print("[WARN] No encuentro runs_index.csv, no puedo hacer cualitativas.")
        else:
            df_runs = None
            if runs_csv and runs_csv.exists():
                df_runs = pd.read_csv(runs_csv)

            if df_runs is None or df_runs.empty:
                print("[WARN] No tengo df_runs (metrics_last_*.csv), no puedo elegir mejores runs por variante.")
            else:
                runs_index = pd.read_csv(runs_index_csv)
                ds_yaml = None
                if "dataset_yaml" in runs_index.columns and len(runs_index) > 0:
                    ds_yaml = Path(str(runs_index.iloc[0]["dataset_yaml"]))
                    ds_yaml = resolve_rel(root, ds_yaml)

                if ds_yaml is None or not ds_yaml.exists():
                    print("[WARN] No puedo resolver dataset_yaml para muestrear val imgs, omito cualitativas.")
                else:
                    render_qualitative_grid(
                        runs_index=runs_index,
                        df_focus_runs=df_runs,
                        variants_order=variants_order,
                        dataset_yaml_for_sampling=ds_yaml,
                        project_root=root,
                        out_dir=plots_dir,
                        n_images=args.qual_n,
                        conf=args.qual_conf,
                        iou=args.qual_iou,
                    )

    print(f"\n[OK] Plots guardados en: {plots_dir}\n")

if __name__ == "__main__":
    main()