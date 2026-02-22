from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")  # evita GUI
import matplotlib.pyplot as plt

from cycler import cycler


def read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


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
        "font.size": 10,
        "font.family": "serif",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "grid.linewidth": 0.7,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "legend.frameon": False,
        "axes.prop_cycle": cycler(color=colors),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def with_color_suffix(p: Path) -> Path:
    if p.stem.endswith("_color"):
        return p
    return p.with_name(p.stem + "_color" + p.suffix)


def color_violinplot_with_inner_box_and_outliers(ax, data, labels, title, ylabel):
    n = len(labels)
    colors = get_color_palette(n)
    positions = list(range(1, n + 1))

    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.88,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=200,
        bw_method="scott",
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_linewidth(1.0)
        body.set_alpha(0.85)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.16,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
        whis=1.5,
    )
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor="black", linewidth=1.2, alpha=0.95)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.1)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.1)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2.0)
    for flier in bp["fliers"]:
        flier.set(
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor="black",
            alpha=0.9,
            linewidth=0.6,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.5, n + 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in_csv", default="xai_scores_gt.csv")
    args = ap.parse_args()

    setup_color_academic_style()

    cfg = read_yaml(Path(args.config))
    root = Path(cfg["project_root"]).resolve()
    meta_dir = root / cfg["paths"]["meta_dir"]

    xai_root = meta_dir / "xai"
    drise_dir = xai_root / "drise_gt"
    out_dir = xai_root / "plots_gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = drise_dir / args.in_csv
    if not in_path.exists():
        raise FileNotFoundError(f"No encuentro {in_path}. Ejecuta xai_drise_gt_yolo.py primero.")

    df = pd.read_csv(in_path)
    if df.empty:
        raise RuntimeError("xai_scores_gt.csv vacío")

    metrics = [
        ("pointing_game", "Pointing Game (↑ mejor)"),
        ("energy_ratio_in_gt", "Energía dentro del GT (↑ mejor)"),
        ("heat_entropy", "Entropía heatmap (↓ mejor)"),
        ("base_score", "Base score IoU×conf (↑ mejor)"),
    ]

    variants = sorted(df["variant"].dropna().unique().tolist())

    for col, pretty in metrics:
        sub = df.dropna(subset=[col]).copy()
        if sub.empty:
            continue

        data = [sub[sub["variant"] == v][col].values for v in variants]

        fig, ax = plt.subplots(figsize=(10, 5))
        color_violinplot_with_inner_box_and_outliers(
            ax, data, variants, f"XAI (GT) — {pretty}", pretty
        )
        fig.tight_layout()

        out_png = with_color_suffix(out_dir / f"xai_gt_{col}_boxplot.png")
        out_pdf = with_color_suffix(out_dir / f"xai_gt_{col}_boxplot.pdf")

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] {out_png}")
        print(f"[OK] {out_pdf}")


if __name__ == "__main__":
    main()