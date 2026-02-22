from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cycler import cycler

def metric_label_keep_en(s: str) -> str:
    s = str(s)
    s = s.replace("metrics/", "").replace("(B)", "")
    s = s.replace("val/", "val ").replace("train/", "train ")
    return s

def es_title_exp4(stage: str, source: str, split: str, tag: str) -> str:
    return f"Exp4 ({stage}/{source}/{split}/{tag})"

def es_title_bars(stage: str, source: str, split: str, tag: str) -> str:
    return f"{es_title_exp4(stage, source, split, tag)} — Data Augmentation policies (media±desv. típ.)"

def es_title_curve(metric: str) -> str:
    return f"Curvas de entrenamiento — {metric_label_keep_en(metric)} (media±desv. típ. entre semillas)"

def es_title_tradeoff(stage: str, source: str, split: str, tag: str, x_m: str, y_m: str) -> str:
    return f"{es_title_exp4(stage, source, split, tag)} — Compromiso ({metric_label_keep_en(x_m)} vs {metric_label_keep_en(y_m)})"

def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def get_stage(cfg: dict, cli_stage: str | None) -> str:
    stage = cli_stage or cfg["exp4"].get("stage", "coarse")
    stages = cfg["exp4"].get("stages", {})
    if stage not in stages:
        raise ValueError(f"Stage '{stage}' no existe en exp4.stages. Disponibles: {list(stages.keys())}")
    return stage

def conf_tag(conf: float | None) -> str:
    if conf is None:
        return "def"
    return f"conf{int(round(conf * 100)):02d}"

def sanitize(s: str) -> str:
    s = s.replace("/", "_").replace("(", "").replace(")", "")
    s = re.sub(r"[^\w\-_\.]", "", s)
    return s

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_color_palette(n: int) -> list:
    """
    Paleta discreta profesional. tab10 hasta 10, tab20 hasta 20,
    y si hay más, hsv discretizado.
    """
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
    """Añade '_color' al final del nombre base (sin extensión)."""
    if out_base.name.endswith("_color"):
        return out_base
    return out_base.with_name(out_base.name + "_color")

def save_fig(fig, out_path: Path, rect=None):
    """
    Guardado consistente + sufijo _color.
    out_path puede venir con .png o sin extensión; lo normalizamos.
    """
    out_base = out_path.with_suffix("")
    out_base = _with_color_suffix(out_base)

    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)

    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

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

def _nice_epoch_ticks(max_epoch: int) -> list[int]:
    if max_epoch <= 10:
        step = 1
    elif max_epoch <= 25:
        step = 5
    elif max_epoch <= 60:
        step = 10
    else:
        step = 10
    ticks = list(range(0, max_epoch + 1, step))
    if ticks and ticks[-1] != max_epoch:
        ticks.append(max_epoch)
    if not ticks:
        ticks = [0]
    return ticks

def load_winner(metrics_dir: Path, source: str, split: str, tag: str) -> Optional[str]:
    winner_json = metrics_dir / f"winner_{source}_{split}_{tag}.json"
    if not winner_json.exists():
        return None
    try:
        w = json.loads(winner_json.read_text(encoding="utf-8"))
        return w.get("winner_variant", None)
    except Exception:
        return None

def order_variants_by_ranking(df_summary: pd.DataFrame, metrics_dir: Path, source: str, split: str, tag: str) -> list[str]:
    ranking_csv = metrics_dir / f"winner_{source}_{split}_{tag}_ranking.csv"
    if not ranking_csv.exists():
        return df_summary["variant"].astype(str).tolist()
    try:
        df_rank = pd.read_csv(ranking_csv)
        if "winner_score" in df_rank.columns:
            df_rank = df_rank.sort_values("winner_score", ascending=False)
        if "variant" in df_rank.columns:
            return df_rank["variant"].astype(str).tolist()
    except Exception:
        pass
    return df_summary["variant"].astype(str).tolist()

def color_barplot_pro(
    means,
    stds,
    labels,
    title,
    ylabel,
    out_path: Path,
    winner_label: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = list(range(len(labels)))

    colors = get_color_palette(len(labels))

    bars = ax.bar(
        x, means,
        yerr=stds, capsize=2.5,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
    )

    for i, b in enumerate(bars):
        if winner_label is not None and str(labels[i]) == str(winner_label):
            b.set_linewidth(1.8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y")

    save_fig(fig, out_path)

def load_results_csv(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "epoch" not in df.columns:
        df["epoch"] = range(len(df))
    return df

def plot_learning_curves_pro(
    df_runs: pd.DataFrame,
    plots_dir: Path,
    stage: str,
    topk_variants: list[str] | None,
    variant_order: list[str] | None = None,
):
    rows = []
    for _, r in df_runs.iterrows():
        variant = str(r.get("variant", ""))
        if topk_variants is not None and variant not in topk_variants:
            continue

        seed = int(r.get("seed", 0))
        run_dir = Path(str(r.get("run_dir", "")))

        df_res = load_results_csv(run_dir)
        if df_res is None:
            print(f"[WARN] No existe results.csv en {run_dir}")
            continue

        df_res = df_res.copy()
        df_res["variant"] = variant
        df_res["seed"] = seed
        rows.append(df_res)

    if not rows:
        print("[WARN] No he podido cargar results.csv para curvas.")
        return

    df_long = pd.concat(rows, ignore_index=True)

    curve_candidates = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]

    available = [c for c in curve_candidates if c in df_long.columns]
    if not available:
        print("[WARN] No encuentro columnas típicas para curvas.")
        print("Columnas:", list(df_long.columns))
        return

    group_cols = ["variant", "epoch"]

    for col in available:
        agg = (
            df_long.groupby(group_cols)[col]
            .agg(["mean", "std"])
            .reset_index()
        )
        
        present_variants = sorted(agg["variant"].astype(str).unique().tolist())
        if variant_order:
            ordered = [v for v in variant_order if v in present_variants]
            missing = [v for v in present_variants if v not in ordered]
            variants = ordered + missing
        else:
            variants = present_variants

        colors = get_color_palette(len(variants))

        fig, ax = plt.subplots(figsize=(6.4, 2.8))

        for i, v in enumerate(variants):
            sub = agg[agg["variant"].astype(str) == str(v)].sort_values("epoch")
            if sub.empty:
                continue

            x = sub["epoch"].astype(int).values
            y = sub["mean"].astype(float).values
            s = sub["std"].astype(float).fillna(0.0).values

            ax.plot(
                x, y,
                color=colors[i],
                linestyle="-",       
                linewidth=1.25,
                label=str(v),
            )

            if len(x) >= 2 and (s > 0).any():
                ax.fill_between(
                    x, y - s, y + s,
                    color=colors[i],
                    alpha=0.18,
                    linewidth=0
                )

        max_epoch = int(agg["epoch"].max()) if len(agg) else 0
        ax.set_xlim(-0.5, max_epoch + 0.5)
        ax.set_xticks(_nice_epoch_ticks(max_epoch))

        ax.set_title(es_title_curve(col))
        ax.set_xlabel("época")
        ax.set_ylabel(metric_label_keep_en(col))
        ax.grid(True)

        ncol = 2 if len(variants) > 3 else 1
        legend_below(ax, ncol=ncol)

        out = plots_dir / f"exp4_{stage}_curve_{sanitize(col)}.png"
        save_fig(fig, out)
        print(f"[OK] {_with_color_suffix(out.with_suffix('')).with_suffix('.png')}")

def plot_tradeoff_scatter_means_exp4(
    df_summary: pd.DataFrame,
    out_path: Path,
    title: str,
    x_metric: str,
    y_metric: str,
    winner_variant: Optional[str] = None,
):
    x_col = f"{x_metric}_mean"
    y_col = f"{y_metric}_mean"
    if x_col not in df_summary.columns or y_col not in df_summary.columns:
        print(f"[WARN] Skip tradeoff: faltan {x_col}/{y_col}")
        return

    d = df_summary.copy()
    d["x"] = d[x_col].astype(float)
    d["y"] = d[y_col].astype(float)
    d = d.dropna(subset=["x", "y"]).copy()
    if d.empty:
        print("[WARN] Tradeoff vacío tras dropna. Skip.")
        return

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    try:
        ax.set_box_aspect(1)
    except Exception:
        ax.set_aspect("equal", adjustable="box")

    colors = get_color_palette(len(d))

    for i, row in enumerate(d.to_dict("records")):
        v = str(row.get("variant", ""))
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

    n = len(d)
    if n <= 6:
        ncol = 2
    elif n <= 10:
        ncol = 3
    else:
        ncol = 4
    legend_below(ax, ncol=ncol)

    save_fig(fig, out_path, rect=[0.0, 0.14, 1.0, 1.0])
    print(f"[OK] {_with_color_suffix(out_path.with_suffix('')).with_suffix('.png')}")

def fmt_pm(mean: float, std: float, nd: int = 3) -> str:
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"

def generate_latex_table(
    metrics_dir: Path,
    stage: str,
    source: str,
    split: str,
    tag: str,
    out_path: Path,
    topk: int = 0,
):
    ranking_csv = metrics_dir / f"winner_{source}_{split}_{tag}_ranking.csv"
    winner_json = metrics_dir / f"winner_{source}_{split}_{tag}.json"
    summary_csv = metrics_dir / f"metrics_summary_{source}_{split}_{tag}.csv"

    if ranking_csv.exists():
        df = pd.read_csv(ranking_csv)
    else:
        if not summary_csv.exists():
            print(f"[WARN] No existe {summary_csv}. No puedo crear tabla.")
            return
        df = pd.read_csv(summary_csv)
        if "winner_score" not in df.columns:
            df["winner_score"] = 0.0
        df = df.sort_values(["winner_score"], ascending=False).reset_index(drop=True)

    winner_variant = str(df.iloc[0]["variant"])
    if winner_json.exists():
        w = json.loads(winner_json.read_text(encoding="utf-8"))
        winner_variant = str(w.get("winner_variant", winner_variant))

    if topk and topk > 0:
        df = df.head(topk).copy()

    metric_order = [
        "metrics/recall(B)",
        "metrics/precision(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    out_rows = []
    for _, r in df.iterrows():
        row = {"Variant": str(r.get("variant", ""))}
        if "winner_score" in df.columns:
            row["Score"] = float(r["winner_score"])

        for m in metric_order:
            m_mean = f"{m}_mean"
            m_std = f"{m}_std"
            if m_mean in df.columns:
                mean = float(r[m_mean]) if not pd.isna(r[m_mean]) else float("nan")
                std = float(r[m_std]) if m_std in df.columns and not pd.isna(r[m_std]) else 0.0
                row[m] = fmt_pm(mean, std, nd=3)

        out_rows.append(row)

    df_out = pd.DataFrame(out_rows)

    for i in range(len(df_out)):
        if str(df_out.iloc[i]["Variant"]) == winner_variant:
            for c in df_out.columns:
                df_out.at[i, c] = f"\\textbf{{{df_out.at[i, c]}}}"

    if "Score" in df_out.columns:
        df_out["Score"] = df_out["Score"].apply(lambda x: x if isinstance(x, str) else f"{x:.4f}")

    rename = {
        "metrics/recall(B)": "Recall (B)",
        "metrics/precision(B)": "Precision (B)",
        "metrics/mAP50(B)": "mAP50 (B)",
        "metrics/mAP50-95(B)": "mAP50-95 (B)",
    }
    df_out = df_out.rename(columns=rename)

    latex = df_out.to_latex(index=False, escape=False, column_format="l" + "c" * (len(df_out.columns) - 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")
    print(f"[OK] Tabla LaTeX guardada en: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None, help="coarse | fine")
    ap.add_argument("--source", default="focus", choices=["focus", "all"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=None)

    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--make_table", action="store_true")
    ap.add_argument("--table_topk", type=int, default=0)

    ap.add_argument("--x_metric", default="metrics/mAP50(B)")
    ap.add_argument("--y_metric", default="metrics/mAP50-95(B)")
    ap.add_argument("--no_paretos", action="store_true")

    args = ap.parse_args()

    setup_color_academic_style()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()

    stage = get_stage(cfg, args.stage)
    meta_dir = (root / cfg["paths"]["meta_dir"] / stage).resolve()

    metrics_dir = ensure_dir(meta_dir / "metrics")
    plots_dir = ensure_dir(meta_dir / "plots")

    tag = conf_tag(args.conf)

    summary_csv = metrics_dir / f"metrics_summary_{args.source}_{args.split}_{tag}.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"No encuentro {summary_csv}. Ejecuta aggregate_exp4_results.py primero.")

    df_sum_raw = pd.read_csv(summary_csv)
    if "variant" not in df_sum_raw.columns:
        raise ValueError(f"{summary_csv} no tiene columna 'variant'")

    winner_variant = load_winner(metrics_dir, args.source, args.split, tag)
    order = order_variants_by_ranking(df_sum_raw, metrics_dir, args.source, args.split, tag)
    df_sum = df_sum_raw.set_index("variant").reindex(order).reset_index()

    labels = df_sum["variant"].astype(str).tolist()

    barplots = [
        ("metrics/recall(B)", "Recall (↑ mejor)"),
        ("metrics/mAP50(B)", "mAP@0.50 (↑ mejor)"),
        ("metrics/mAP50-95(B)", "mAP@0.50:0.95 (↑ mejor)"),
        ("metrics/precision(B)", "Precision (↑ mejor)"),
    ]

    for m, pretty in barplots:
        m_mean = f"{m}_mean"
        m_std = f"{m}_std"
        if m_mean not in df_sum.columns:
            continue

        means = df_sum[m_mean].astype(float).values
        stds = df_sum[m_std].fillna(0.0).astype(float).values if m_std in df_sum.columns else [0.0] * len(means)

        out_path = plots_dir / f"exp4_{stage}_{args.source}_{args.split}_{tag}_{sanitize(m)}_bar.png"
        color_barplot_pro(
            means=means,
            stds=stds,
            labels=labels,
            title=es_title_bars(stage, args.source, args.split, tag),
            ylabel=pretty,
            out_path=out_path,
            winner_label=winner_variant,
        )
        print(f"[OK] {_with_color_suffix(out_path.with_suffix('')).with_suffix('.png')}")

    runs_index = meta_dir / "runs" / "runs_index.csv"
    if runs_index.exists():
        df_runs = pd.read_csv(runs_index)
        topk_variants = None
        if args.topk and args.topk > 0:
            rank_csv = metrics_dir / f"winner_{args.source}_{args.split}_{tag}_ranking.csv"
            if rank_csv.exists():
                df_rank = pd.read_csv(rank_csv)
                topk_variants = df_rank["variant"].head(args.topk).astype(str).tolist()
                print(f"[INFO] Curvas solo top-{args.topk}: {topk_variants}")
        plot_learning_curves_pro(
            df_runs=df_runs,
            plots_dir=plots_dir,
            stage=stage,
            topk_variants=topk_variants,
            variant_order=labels, 
        )
    else:
        print(f"[WARN] No encuentro {runs_index}. No puedo dibujar curvas.")

    if not args.no_paretos:
        df_pareto = df_sum
        if args.topk and args.topk > 0:
            df_pareto = df_sum.head(args.topk).copy()
            print(f"[INFO] Paretos solo top-{args.topk}: {df_pareto['variant'].astype(str).tolist()}")

        out_path = plots_dir / f"exp4_{stage}_{args.source}_{args.split}_{tag}_tradeoff_custom.png"  # ✅ FIX exp4
        plot_tradeoff_scatter_means_exp4(
            df_summary=df_pareto,
            out_path=out_path,
            title=es_title_tradeoff(stage, args.source, args.split, tag, args.x_metric, args.y_metric),
            x_metric=args.x_metric,
            y_metric=args.y_metric,
            winner_variant=winner_variant,
        )

        tradeoff_pairs = [
            ("metrics/mAP50(B)", "metrics/mAP50-95(B)"),
            ("metrics/mAP50(B)", "metrics/recall(B)"),
            ("metrics/mAP50-95(B)", "metrics/recall(B)"),
            ("metrics/precision(B)", "metrics/recall(B)"),
            ("metrics/mAP50(B)", "metrics/precision(B)"),
        ]
        for x_m, y_m in tradeoff_pairs:
            out_path = plots_dir / f"exp4_{stage}_{args.source}_{args.split}_{tag}_tradeoff_{sanitize(x_m)}_vs_{sanitize(y_m)}.png"  # ✅ FIX exp4
            plot_tradeoff_scatter_means_exp4(
                df_summary=df_pareto,
                out_path=out_path,
                title=es_title_tradeoff(stage, args.source, args.split, tag, x_m, y_m),
                x_metric=x_m,
                y_metric=y_m,
                winner_variant=winner_variant,
            )

    if args.make_table:
        out_tex = metrics_dir / f"exp4_{stage}_{args.source}_{args.split}_{tag}_table.tex"
        generate_latex_table(
            metrics_dir=metrics_dir,
            stage=stage,
            source=args.source,
            split=args.split,
            tag=tag,
            out_path=out_tex,
            topk=args.table_topk,
        )

    print(f"\n[OK] Plots guardados en: {plots_dir}\n")


if __name__ == "__main__":
    main()
