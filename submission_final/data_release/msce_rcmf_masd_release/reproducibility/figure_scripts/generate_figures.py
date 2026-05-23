from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DIAG_ROOT = ROOT / "outputs" / "exp" / "diagnostics"
DEFAULT_DIAG_100_NAME = "masd_final_trisoup_unionmask_clean_100run"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "figure.dpi": 300,
        "axes.linewidth": 0.9,
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.35,
        "grid.color": "#9AA1A8",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
    }
)


PALETTE = {
    "bg":          "#FFFFFF",
    "ink":         "#1A1A1A",   # near-black text
    "line":        "#CCCCCC",   # dividers / subtle borders
    "blue":        "#1B4F8A",   # primary deep blue
    "blue_m":      "#2E75B6",   # medium blue (graph branch)
    "blue_soft":   "#D6E8F7",   # primary light fill
    "blue_ll":     "#EAF3FB",   # very light fill / stripes
    "orange":      "#C05A16",   # warm orange accent
    "orange_soft": "#FAE5D3",   # orange light fill
    "green":       "#1A6B3C",   # dark green (positive / improvement)
    "green_soft":  "#D5EDDF",   # green light fill
    "slate":       "#5B7FA6",   # slate medium blue (secondary)
    "slate_soft":  "#EAF3FB",   # slate light fill
    "gray":        "#555555",   # body annotation text
    "red":         "#8B1A1A",   # dark red (negative / regression)
    "red_soft":    "#F5DADA",   # red light fill
    "teal":        "#5B7FA6",   # alias → slate (no separate teal)
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=360, bbox_inches="tight")
    plt.close(fig)


def resolve_diag_100(diag_dir: str | None = None) -> Path:
    if diag_dir:
        path = Path(diag_dir)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"diagnostic directory does not exist: {path}")
        return path

    preferred = DIAG_ROOT / DEFAULT_DIAG_100_NAME
    if preferred.exists():
        return preferred

    candidates = sorted(
        path
        for path in DIAG_ROOT.glob("*final*trisoup*unionmask*100run")
        if path.is_dir() and path.name.endswith("unionmask_100run")
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"could not locate a final 100-run diagnostic package under {DIAG_ROOT}; "
            f"expected {preferred.name}"
        )
    raise RuntimeError(
        "multiple candidate 100-run diagnostic packages were found; "
        "pass --diag-dir explicitly to select one"
    )


def parse_ci(text: str) -> tuple[float, float]:
    vals = re.findall(r"[-+]?\d+(?:\.\d+)?", str(text))
    if len(vals) < 2:
        raise ValueError(f"Cannot parse CI from {text!r}")
    return float(vals[0]), float(vals[1])


def clean_label(label: str) -> str:
    mapping = {
        "Main test set": "Main test",
        "Hard subgroup": "Hard subgroup",
        "External holdout": "External holdout",
        "aromatic_dense": "Aromatic dense",
        "ester_or_carbonate": "Ester / carbonate",
        "fluorinated": "Fluorinated",
        "sulfone": "Sulfone",
        "amide": "Amide",
        "ether_oxygen": "Ether oxygen",
        "imide_like": "Imide-like",
        "other": "Other",
    }
    return mapping.get(label, label)


def _legacy_fig_overview_unused() -> None:
    fig, ax = plt.subplots(figsize=(15.6, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "ink": "#1F2933",
        "muted": "#5B6670",
        "line": "#C9CED6",
        "panel": "#FAF8F5",
        "blue": "#4C78A8",
        "blue_soft": "#EAF2F8",
        "green": "#2A9D8F",
        "green_soft": "#E8F5F1",
        "orange": "#E39D26",
        "orange_soft": "#FCF2DD",
        "red": "#B6413D",
        "red_soft": "#F8E7E4",
        "gray": "#9AA0A6",
        "gray_soft": "#F3F4F6",
    }

    def rounded_box(
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fc: str = "#FFFFFF",
        ec: str | None = None,
        lw: float = 1.2,
        radius: float = 0.015,
        alpha: float = 1.0,
        zorder: int = 1,
    ) -> patches.FancyBboxPatch:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={radius}",
            linewidth=lw,
            edgecolor=ec or colors["line"],
            facecolor=fc,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(rect)
        return rect

    def text(
        x: float,
        y: float,
        s: str,
        *,
        size: float = 10.0,
        weight: str | None = None,
        color: str | None = None,
        ha: str = "center",
        va: str = "center",
        style: str | None = None,
        zorder: int = 5,
    ) -> None:
        ax.text(
            x,
            y,
            s,
            fontsize=size,
            fontweight=weight,
            color=color or colors["ink"],
            ha=ha,
            va=va,
            style=style,
            zorder=zorder,
        )

    def arrow(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str,
        lw: float = 1.6,
        style: str = "-|>",
        ms: float = 13,
        zorder: int = 4,
    ) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle=style,
                lw=lw,
                color=color,
                mutation_scale=ms,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=zorder,
        )

    def elbow_arrow(
        x1: float,
        y1: float,
        xm: float,
        y2: float,
        x2: float,
        *,
        color: str,
        lw: float = 1.5,
        ms: float = 12,
        zorder: int = 4,
    ) -> None:
        ax.plot([x1, xm, xm, x2], [y1, y1, y2, y2], color=color, lw=lw, zorder=zorder)
        arrow(x2 - 0.0001, y2, x2, y2, color=color, lw=lw, ms=ms, zorder=zorder)

    def header_rule(x: float, w: float, color: str = colors["line"]) -> None:
        ax.plot([x, x + w], [0.85, 0.85], color=color, lw=1.0, zorder=1)

    def descriptor_icon(x: float, y: float, w: float, h: float, color: str) -> None:
        rounded_box(x, y, w, h, fc="#FFFFFF", ec=color, lw=1.1, radius=0.01, zorder=3)
        cols, rows = 5, 4
        gx0, gy0 = x + 0.10 * w, y + 0.18 * h
        gw, gh = 0.74 * w, 0.62 * h
        ax.plot([gx0, gx0 + gw], [gy0 + gh, gy0 + gh], color=color, lw=1.0, zorder=4)
        for i in range(cols + 1):
            xx = gx0 + gw * i / cols
            ax.plot([xx, xx], [gy0, gy0 + gh], color="#B7C7D9", lw=0.9, zorder=4)
        for j in range(rows + 1):
            yy = gy0 + gh * j / rows
            ax.plot([gx0, gx0 + gw], [yy, yy], color="#B7C7D9", lw=0.9, zorder=4)
        for i in range(2):
            for j in range(2):
                rounded_box(
                    gx0 + 0.02 * gw + i * 0.18 * gw,
                    gy0 + 0.58 * gh - j * 0.23 * gh,
                    0.10 * gw,
                    0.10 * gh,
                    fc=color,
                    ec=color,
                    lw=0.0,
                    radius=0.004,
                    zorder=5,
                )

    def graph_icon(x: float, y: float, w: float, h: float, color: str) -> None:
        nodes = np.array(
            [
                [0.15, 0.45],
                [0.34, 0.68],
                [0.50, 0.50],
                [0.69, 0.65],
                [0.84, 0.42],
                [0.53, 0.25],
            ]
        )
        edges = [(0, 1), (1, 2), (2, 3), (2, 5), (3, 4), (0, 2), (4, 5)]
        for i, j in edges:
            ax.plot(
                [x + w * nodes[i, 0], x + w * nodes[j, 0]],
                [y + h * nodes[i, 1], y + h * nodes[j, 1]],
                color=color,
                lw=1.6,
                zorder=4,
            )
        atom_colors = ["#C4DFAA", "#A8DADC", "#E9C46A", "#F4A261", "#9CCB86", "#BDE0FE"]
        for idx, (nx, ny) in enumerate(nodes):
            ax.add_patch(
                patches.Circle(
                    (x + w * nx, y + h * ny),
                    radius=min(w, h) * 0.075,
                    facecolor=atom_colors[idx % len(atom_colors)],
                    edgecolor=color,
                    lw=1.0,
                    zorder=5,
                )
            )

    def bars_icon(x: float, y: float, w: float, h: float, color: str) -> None:
        heights = [0.28, 0.45, 0.65, 0.88]
        fills = [colors["blue_soft"], colors["orange_soft"], colors["green_soft"], "#F4D7A6"]
        for idx, hh in enumerate(heights):
            bw = 0.14 * w
            bx = x + 0.10 * w + idx * 0.19 * w
            rounded_box(
                bx,
                y + 0.12 * h,
                bw,
                hh * 0.70 * h,
                fc=fills[idx],
                ec=color,
                lw=1.0,
                radius=0.004,
                zorder=4,
            )

    def polymer_chain(x: float, y: float, w: float, h: float, color: str, *, zorder: int = 4) -> None:
        xs = np.linspace(x + 0.06 * w, x + 0.94 * w, 12)
        ys = y + 0.50 * h + 0.16 * h * np.sin(np.linspace(0, 3.6 * np.pi, xs.size))
        ax.plot(xs, ys, color=color, lw=1.8, zorder=zorder)
        fill_cycle = ["#F1DCA7", "#BFE3E1", "#F0C6A8", "#D8E8B5"]
        for idx, (xx, yy) in enumerate(zip(xs, ys)):
            ax.add_patch(
                patches.Circle(
                    (xx, yy),
                    radius=min(w, h) * 0.040,
                    facecolor=fill_cycle[idx % len(fill_cycle)],
                    edgecolor=color,
                    lw=0.9,
                    zorder=zorder + 1,
                )
            )

    def modality_card(
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        title: str,
        notation: str,
        subtitle: str,
        line_color: str,
        fill: str,
        icon_kind: str,
        footer: list[str] | None = None,
    ) -> None:
        rounded_box(x, y, w, h, fc=fill, ec=line_color, lw=1.4, radius=0.018, zorder=2)
        rounded_box(x + 0.006, y + h - 0.055, 0.40 * w, 0.045, fc=line_color, ec=line_color, lw=0.0, radius=0.012, zorder=3)
        text(x + 0.022, y + h - 0.032, title, size=10.6, weight="bold", color="#FFFFFF", ha="left")
        text(x + 0.47 * w, y + h - 0.060, notation, size=11.0, weight="bold", color=colors["ink"], ha="left")
        text(x + 0.47 * w, y + h - 0.098, subtitle, size=9.3, color=colors["ink"], ha="left")
        icon_x, icon_y, icon_w, icon_h = x + 0.018, y + 0.032, 0.30 * w, h - 0.094
        if icon_kind == "descriptor":
            descriptor_icon(icon_x, icon_y, icon_w, icon_h, line_color)
        elif icon_kind == "graph":
            rounded_box(icon_x, icon_y, icon_w, icon_h, fc="#FFFFFF", ec=line_color, lw=1.1, radius=0.010, zorder=3)
            graph_icon(icon_x + 0.03 * icon_w, icon_y + 0.06 * icon_h, 0.94 * icon_w, 0.84 * icon_h, line_color)
        elif icon_kind == "context":
            text(x + 0.018, y + h - 0.103, subtitle, size=9.1, color=colors["ink"], ha="left")
            rounded_box(icon_x, y + 0.085 * h, w - 0.036, 0.30 * h, fc="#FFF9EE", ec="#E8C58D", lw=0.9, radius=0.010, zorder=3)
            polymer_chain(x + 0.03 * w, y + 0.155 * h, 0.90 * w, 0.16 * h, line_color, zorder=4)
            if footer:
                foot_y = y + 0.02
                item_w = (w - 0.050) / len(footer)
                for idx, label in enumerate(footer):
                    bx = x + 0.012 + idx * item_w
                    rounded_box(
                        bx,
                        foot_y,
                        item_w - 0.010,
                        0.065,
                        fc="#FFF6E6",
                        ec="#E8C58D",
                        lw=0.9,
                        radius=0.010,
                        zorder=3,
                    )
                    if "chain" in label.lower():
                        graph_icon(bx + 0.012, foot_y + 0.016, 0.055, 0.032, "#C4902A")
                    elif "segment" in label.lower():
                        descriptor_icon(bx + 0.012, foot_y + 0.013, 0.055, 0.036, "#C4902A")
                    elif "neigh" in label.lower():
                        graph_icon(bx + 0.010, foot_y + 0.014, 0.060, 0.034, "#C4902A")
                    else:
                        bars_icon(bx + 0.010, foot_y + 0.010, 0.060, 0.040, "#C4902A")
                    text(bx + 0.075, foot_y + 0.032, label, size=7.8, ha="left")

    fig.patch.set_facecolor(colors["panel"])
    ax.plot([0.03, 0.97], [0.93, 0.93], color=colors["line"], lw=1.1)
    text(0.50, 0.965, "Overall architecture of MSCE-RCMF-MASD", size=19.0, weight="bold")

    panels = {
        "p1": (0.03, 0.18, 0.16, 0.67),
        "p2": (0.205, 0.18, 0.16, 0.67),
        "p3": (0.380, 0.18, 0.22, 0.67),
        "p4": (0.615, 0.18, 0.17, 0.67),
        "p5": (0.800, 0.18, 0.17, 0.67),
    }
    titles = {
        "p1": "1. Inputs",
        "p2": "2. Branch Encoders\nand Branch Heads",
        "p3": "3. MSCE: Hierarchical\nContext Selection",
        "p4": "4. RCMF: Reliability-Conditioned\nFusion",
        "p5": "5. MASD: Bounded Structured\nCorrection",
    }

    for key, (x, y, w, h) in panels.items():
        ax.plot([x, x + w], [0.82, 0.82], color=colors["line"], lw=0.95)
        text(x + w / 2, 0.855, titles[key], size=11.0, weight="bold")
        if key != "p5":
            ax.plot([x + w + 0.007, x + w + 0.007], [0.18, 0.82], color="#D7DADF", lw=1.0, linestyle=(0, (3, 3)))

    p1x, _, p1w, _ = panels["p1"]
    modality_card(
        p1x + 0.005,
        0.60,
        p1w - 0.010,
        0.16,
        title="Descriptor view",
        notation=r"$x_d \in \mathbb{R}^{528}$",
        subtitle="global handcrafted descriptors",
        line_color=colors["blue"],
        fill=colors["blue_soft"],
        icon_kind="descriptor",
    )
    modality_card(
        p1x + 0.005,
        0.40,
        p1w - 0.010,
        0.15,
        title="Graph view",
        notation=r"$G=(V,E)$",
        subtitle="molecular graph",
        line_color=colors["green"],
        fill=colors["green_soft"],
        icon_kind="graph",
    )
    modality_card(
        p1x + 0.005,
        0.15,
        p1w - 0.010,
        0.23,
        title="Polymer context",
        notation=r"$c \in \mathbb{R}^{980}$",
        subtitle="multiscale polymer context",
        line_color=colors["orange"],
        fill=colors["orange_soft"],
        icon_kind="context",
        footer=["Chain-level", "Segment-window", "Graph-neighborhood", "Statistics"],
    )

    p2x, _, p2w, _ = panels["p2"]
    rounded_box(p2x + 0.02, 0.65, p2w - 0.04, 0.055, fc=colors["blue_soft"], ec=colors["blue"], lw=1.3)
    text(p2x + p2w / 2, 0.677, "Descriptor encoder", size=10.8, weight="bold", color=colors["blue"])
    rounded_box(p2x + 0.05, 0.585, 0.055, 0.040, fc=colors["blue"], ec=colors["blue"], lw=0.0)
    text(p2x + 0.078, 0.605, r"$h_d$", size=11.3, weight="bold", color="#FFFFFF")
    rounded_box(p2x + 0.12, 0.57, p2w - 0.14, 0.050, fc=colors["blue_soft"], ec=colors["blue"], lw=1.2)
    text(p2x + 0.12 + (p2w - 0.14) / 2, 0.595, r"Head: $y_d, u_d$", size=10.0, weight="bold", color=colors["blue"])

    rounded_box(p2x + 0.02, 0.42, p2w - 0.04, 0.055, fc=colors["green_soft"], ec=colors["green"], lw=1.3)
    text(p2x + p2w / 2, 0.447, "Graph encoder", size=10.8, weight="bold", color=colors["green"])
    rounded_box(p2x + 0.05, 0.355, 0.055, 0.040, fc=colors["green"], ec=colors["green"], lw=0.0)
    text(p2x + 0.078, 0.375, r"$h_g$", size=11.3, weight="bold", color="#FFFFFF")
    rounded_box(p2x + 0.12, 0.34, p2w - 0.14, 0.050, fc=colors["green_soft"], ec=colors["green"], lw=1.2)
    text(p2x + 0.12 + (p2w - 0.14) / 2, 0.365, r"Head: $y_g, u_g$", size=10.0, weight="bold", color=colors["green"])

    arrow(p1x + p1w - 0.005, 0.68, p2x + 0.02, 0.677, color=colors["blue"])
    arrow(p1x + p1w - 0.005, 0.475, p2x + 0.02, 0.447, color=colors["green"])
    arrow(p2x + p2w / 2, 0.65, p2x + 0.078, 0.625, color=colors["blue"])
    arrow(p2x + 0.105, 0.605, p2x + 0.12, 0.595, color=colors["blue"], ms=10)
    arrow(p2x + p2w / 2, 0.42, p2x + 0.078, 0.395, color=colors["green"])
    arrow(p2x + 0.105, 0.375, p2x + 0.12, 0.365, color=colors["green"], ms=10)

    p3x, _, p3w, _ = panels["p3"]
    rounded_box(p3x + 0.075, 0.73, p3w - 0.15, 0.06, fc=colors["orange"], ec=colors["orange"], lw=0.0)
    text(p3x + p3w / 2, 0.760, "Context split", size=12.0, weight="bold", color="#FFFFFF")
    scale_y = [0.645, 0.585, 0.525, 0.465]
    scale_labels = [
        r"Scale 1: Chain-level n-gram  $\varphi_1$",
        r"Scale 2: Segment-window  $\varphi_2$",
        r"Scale 3: Graph-neighborhood  $\varphi_3$",
        r"Scale 4: Interpretable polymer statistics  $\varphi_4$",
    ]
    for yy, label in zip(scale_y, scale_labels):
        rounded_box(p3x + 0.050, yy, p3w - 0.100, 0.050, fc="#FFF9EE", ec="#E0B86D", lw=1.0, radius=0.012)
        text(p3x + 0.065, yy + 0.025, label, size=9.8, ha="left")
    rounded_box(p3x + 0.060, 0.375, p3w - 0.120, 0.065, fc=colors["orange"], ec=colors["orange"], lw=0.0)
    text(p3x + p3w / 2, 0.407, r"Top-$k$ context gate $(k=3)$", size=11.2, weight="bold", color="#FFFFFF")
    rounded_box(p3x + 0.050, 0.285, p3w - 0.100, 0.060, fc="#FFF8E9", ec="#E0B86D", lw=1.0)
    text(p3x + p3w / 2, 0.315, r"Selected context embedding $h_{ctx}$", size=11.0, weight="bold")
    text(p3x + p3w / 2, 0.220, "MSCE selects useful context scales before fusion.", size=10.0, style="italic", color=colors["muted"])

    arrow(p1x + p1w - 0.005, 0.265, p3x + 0.075, 0.760, color=colors["orange"])
    arrow(p3x + p3w / 2, 0.73, p3x + p3w / 2, 0.695, color=colors["orange"])
    arrow(p3x + p3w / 2, 0.465, p3x + p3w / 2, 0.440, color=colors["orange"])
    arrow(p3x + p3w / 2, 0.375, p3x + p3w / 2, 0.345, color=colors["orange"])

    p4x, _, p4w, _ = panels["p4"]
    rounded_box(p4x + 0.018, 0.68, p4w - 0.036, 0.11, fc="#FFFFFF", ec=colors["gray"], lw=1.0)
    text(p4x + p4w / 2, 0.765, r"Reliability vector $q$", size=12.0, weight="bold")
    text(p4x + p4w / 2, 0.715, r"$q=[y_d,\, y_g,\, u_d,\, u_g,\, |y_d-y_g|]$", size=10.4)
    text(p4x + p4w / 2, 0.690, "from descriptor and graph branch heads", size=8.6, color=colors["muted"], style="italic")

    rounded_box(p4x + 0.020, 0.505, p4w - 0.040, 0.090, fc=colors["red"], ec=colors["red"], lw=0.0)
    text(
        p4x + p4w / 2,
        0.550,
        "Anchor-centered\nreliability-conditioned fusion",
        size=11.8,
        weight="bold",
        color="#FFFFFF",
    )
    text(p4x + p4w / 2, 0.500, r"inputs: $h_d,\ h_g,\ h_{ctx},\ q$", size=9.0, color="#FDEEEB")

    rounded_box(p4x + 0.035, 0.40, p4w - 0.070, 0.060, fc=colors["red_soft"], ec=colors["red"], lw=1.2)
    text(p4x + p4w / 2, 0.430, r"Fused representation $h_{fuse}$", size=10.6, weight="bold", color=colors["red"])
    rounded_box(p4x + 0.050, 0.31, p4w - 0.100, 0.055, fc=colors["red_soft"], ec=colors["red"], lw=1.2)
    text(p4x + p4w / 2, 0.337, r"Anchor prediction $y_{anchor}$", size=10.5, weight="bold", color=colors["red"])

    text(p4x + p4w / 2, 0.205, "RCMF uses reliability to regulate cross-view interaction.", size=10.0, style="italic", color=colors["muted"])

    arrow(p4x + p4w / 2, 0.68, p4x + p4w / 2, 0.595, color=colors["red"])
    arrow(p3x + p3w - 0.05, 0.315, p4x + 0.02, 0.548, color=colors["orange"], lw=1.4)
    arrow(p4x + p4w / 2, 0.68, p4x + p4w / 2, 0.595, color=colors["red"])
    arrow(p4x + p4w / 2, 0.505, p4x + p4w / 2, 0.460, color=colors["red"])
    arrow(p4x + p4w / 2, 0.40, p4x + p4w / 2, 0.365, color=colors["red"])

    p5x, _, p5w, _ = panels["p5"]
    rounded_box(p5x + 0.025, 0.73, p5w - 0.050, 0.045, fc=colors["gray_soft"], ec=colors["gray"], lw=1.0)
    text(p5x + p5w / 2, 0.752, "Signed correction slots", size=10.8, weight="bold")
    slot_y = [0.635, 0.565, 0.495, 0.425]
    slot_labels = [r"$\Delta_1(+)$", r"$\Delta_2(+)$", r"$\Delta_3(-)$", r"$\Delta_4(-)$"]
    for yy, label in zip(slot_y, slot_labels):
        rounded_box(p5x + 0.045, yy, p5w - 0.090, 0.050, fc="#FFF3F0", ec=colors["red"], lw=1.1, radius=0.012)
        text(p5x + p5w / 2, yy + 0.025, label, size=12.0, weight="bold", color=colors["red"])
    rounded_box(p5x + 0.055, 0.315, p5w - 0.110, 0.050, fc=colors["gray_soft"], ec=colors["gray"], lw=1.0)
    text(p5x + p5w / 2, 0.340, r"Sparsemax $\alpha$", size=10.4, weight="bold")
    rounded_box(p5x + 0.055, 0.245, p5w - 0.110, 0.050, fc=colors["gray_soft"], ec=colors["gray"], lw=1.0)
    text(p5x + p5w / 2, 0.270, r"Gate $g$", size=10.4, weight="bold")
    rounded_box(p5x + 0.035, 0.170, p5w - 0.070, 0.058, fc="#FFF3F0", ec=colors["red"], lw=1.1)
    text(p5x + p5w / 2, 0.199, r"$\hat{y}=y_{anchor}+g\sum_m \alpha_m \Delta_m$", size=10.1)
    rounded_box(p5x + 0.020, 0.095, p5w - 0.040, 0.055, fc="#FCEAE8", ec=colors["red"], lw=1.1)
    text(p5x + p5w / 2, 0.122, "Final $T_g$ estimate\n$\\hat{y}$", size=10.4, weight="bold", color=colors["red"])
    text(p5x + p5w / 2, 0.160, "MASD applies sparse signed and bounded correction.", size=9.8, style="italic", color=colors["muted"], va="top")

    arrow(p4x + p4w - 0.015, 0.430, p5x + 0.045, 0.660, color=colors["red"], lw=1.3)
    arrow(p4x + p4w - 0.015, 0.337, p5x + 0.035, 0.199, color=colors["red"], lw=1.3)
    arrow(p5x + p5w / 2, 0.73, p5x + p5w / 2, 0.685, color=colors["red"])
    arrow(p5x + p5w / 2, 0.425, p5x + p5w / 2, 0.365, color=colors["red"])
    arrow(p5x + p5w / 2, 0.315, p5x + p5w / 2, 0.295, color=colors["red"])
    arrow(p5x + p5w / 2, 0.245, p5x + p5w / 2, 0.228, color=colors["red"])
    arrow(p5x + p5w / 2, 0.170, p5x + p5w / 2, 0.150, color=colors["red"])

    rounded_box(0.03, 0.035, 0.94, 0.095, fc="#FFFFFF", ec=colors["gray"], lw=1.0, radius=0.012, alpha=0.95)
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.38, 0.083),
            0.24,
            0.038,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            linewidth=1.0,
            edgecolor=colors["gray"],
            facecolor=colors["gray_soft"],
        )
    )
    text(0.50, 0.102, "Training-only auxiliary", size=11.3, weight="bold")
    rounded_box(0.43, 0.052, 0.14, 0.025, fc="#FFFFFF", ec=colors["gray"], lw=0.9, radius=0.008)
    text(0.50, 0.064, r"Geometry prior $z_{geo}\in\mathbb{R}^{11}$", size=10.0)
    text(0.50, 0.042, "training-only auxiliary; not used at inference", size=9.7, color=colors["muted"], style="italic")

    save(fig, "fig1_overview")


def fig_main_results(diag_dir: Path) -> None:
    import json as _json

    main_df = pd.read_csv(diag_dir / "main_results_table.csv")
    subgroup_df = pd.read_csv(diag_dir / "subgroup_results_table.csv")
    improvement_df = pd.read_csv(diag_dir / "improvement_table.csv")

    baseline_main = float(main_df.iloc[0]["MAE (K)"])
    final_main = float(main_df.iloc[1]["MAE (K)"])
    baseline_hard = float(subgroup_df.iloc[0]["Hard subgroup MAE (K)"])
    final_hard = float(subgroup_df.iloc[1]["Hard subgroup MAE (K)"])
    baseline_external = float(subgroup_df.iloc[0]["External holdout MAE (K)"])
    final_external = float(subgroup_df.iloc[1]["External holdout MAE (K)"])

    split_rows = [
        ("Main test set", baseline_main, final_main),
        ("Hard subgroup", baseline_hard, final_hard),
        ("External holdout", baseline_external, final_external),
    ]
    relative_reductions = [
        100.0 * (baseline_val - final_val) / baseline_val
        for _, baseline_val, final_val in split_rows
    ]
    reduction_df = improvement_df[
        improvement_df["Evaluation split"].isin(["Main test set", "Hard subgroup", "External holdout"])
    ].copy()
    reduction_df["label"] = reduction_df["Evaluation split"].map(clean_label)
    reduction_df["ci_low"], reduction_df["ci_high"] = zip(*reduction_df["95% CI"].map(parse_ci))
    reduction_df["order"] = reduction_df["Evaluation split"].map(
        {"Main test set": 0, "Hard subgroup": 1, "External holdout": 2}
    )
    reduction_df = reduction_df.sort_values("order").reset_index(drop=True)

    # Per-seed data for Panel C
    with open(diag_dir / "stats.json") as f:
        stats = _json.load(f)
    psr = stats["per_seed_records"]
    per_seed = {
        "Main test set":    np.array([r["primary_mae_reduction_k"]  for r in psr], dtype=float),
        "Hard subgroup":    np.array([r["hard_mae_reduction_k"]     for r in psr], dtype=float),
        "External holdout": np.array([r["external_mae_reduction_k"] for r in psr], dtype=float),
    }

    # ── Layout: 3 panels ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.0, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.95, 1.25], wspace=0.32)
    ax_left   = fig.add_subplot(gs[0, 0])
    ax_mid    = fig.add_subplot(gs[0, 1])
    ax_right  = fig.add_subplot(gs[0, 2])
    fig.patch.set_facecolor(PALETTE["bg"])

    split_labels = [clean_label(n) for n, _, _ in split_rows]
    y = np.arange(len(split_rows))
    stripe_color = PALETTE["blue_ll"]
    for ax_ in (ax_left, ax_mid):
        for yi in y:
            if yi % 2 == 0:
                ax_.axhspan(yi - 0.45, yi + 0.45, color=stripe_color, zorder=0)

    # ── Panel A: lollipop MAE ─────────────────────────────────────────────────
    baseline_color = "#888888"
    final_color = PALETTE["orange"]
    connector_color = "#CCCCCC"

    for yi, ((_, baseline_val, final_val), rel_pct) in enumerate(zip(split_rows, relative_reductions)):
        ax_left.plot([baseline_val, final_val], [yi, yi], color=connector_color, lw=2.6, zorder=1)
        ax_left.scatter(baseline_val, yi, s=90, color=baseline_color, edgecolor="white", linewidth=1.0, zorder=3)
        is_hard = (yi == 1)
        ax_left.scatter(
            final_val, yi,
            s=115 if is_hard else 90,
            color=final_color,
            edgecolor=PALETTE["ink"],
            linewidth=0.7,
            zorder=4,
        )
        ax_left.text(baseline_val - 0.08, yi - 0.20, f"{baseline_val:.2f}",
                     fontsize=8.5, color=PALETTE["gray"], ha="right")
        ax_left.text(final_val + 0.08, yi + 0.20, f"{final_val:.2f}",
                     fontsize=8.5, color=PALETTE["ink"], ha="left")
        rel_color = PALETTE["orange"] if is_hard else PALETTE["gray"]
        fw = "bold" if is_hard else "normal"
        ax_left.text(
            0.5 * (baseline_val + final_val),
            yi - (0.26 if is_hard else 0.18),
            f"{rel_pct:.1f}%",
            fontsize=8.2, fontweight=fw, color=rel_color,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor=rel_color, linewidth=0.85, alpha=0.96),
        )

    ax_left.set_yticks(y)
    ax_left.set_yticklabels(split_labels)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("MAE (K)")
    ax_left.set_title("Aggregate MAE: baseline vs. proposed", fontsize=11.0, pad=9)
    ax_left.grid(axis="x", linestyle=":", alpha=0.28)
    for sp in ("top", "right", "left"):
        ax_left.spines[sp].set_visible(False)
    ax_left.tick_params(axis="y", length=0)
    ax_left.text(0.02, 1.05, "A", transform=ax_left.transAxes, fontsize=12.5, fontweight="bold")
    ax_left.scatter([], [], s=75, color=baseline_color, label="Baseline (Simple Concat)")
    ax_left.scatter([], [], s=75, color=final_color, edgecolor=PALETTE["ink"],
                    linewidth=0.7, label="MSCE-RCMF-MASD")
    ax_left.legend(loc="lower right", frameon=False, fontsize=8.5)

    # ── Panel B: CI errorbar ──────────────────────────────────────────────────
    reduction_vals = reduction_df["MAE reduction (K)"].to_numpy(dtype=float)
    ci_low  = reduction_df["ci_low"].to_numpy(dtype=float)
    ci_high = reduction_df["ci_high"].to_numpy(dtype=float)
    err_low  = reduction_vals - ci_low
    err_high = ci_high - reduction_vals
    point_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]

    ax_mid.axvline(0, color=PALETTE["ink"], lw=1.0, linestyle="--", zorder=1)
    for yi in y:
        if yi % 2 == 0:
            ax_mid.axhspan(yi - 0.45, yi + 0.45, color=stripe_color, zorder=0)
    ax_mid.errorbar(
        reduction_vals, y,
        xerr=[err_low, err_high],
        fmt="none", ecolor=PALETTE["ink"], elinewidth=1.4, capsize=4, zorder=2,
    )
    ax_mid.scatter(reduction_vals, y, s=[90, 120, 90],
                   color=point_colors, edgecolor="white", linewidth=1.0, zorder=3)
    for yi, value, lo, hi in zip(y, reduction_vals, ci_low, ci_high):
        ax_mid.text(hi + 0.06, yi,
                    f"{value:.2f}\n[{lo:.2f}, {hi:.2f}]",
                    va="center", ha="left", fontsize=8.2, color=PALETTE["ink"])

    ax_mid.set_yticks(y)
    ax_mid.set_yticklabels(reduction_df["label"].tolist())
    ax_mid.invert_yaxis()
    ax_mid.set_xlabel("MAE reduction (K)")
    ax_mid.set_title("Mean reduction (95% CI, 100 runs)", fontsize=11.0, pad=9)
    ax_mid.grid(axis="x", linestyle=":", alpha=0.28)
    for sp in ("top", "right", "left"):
        ax_mid.spines[sp].set_visible(False)
    ax_mid.tick_params(axis="y", length=0)
    ax_mid.text(0.02, 1.05, "B", transform=ax_mid.transAxes, fontsize=12.5, fontweight="bold")

    # ── Panel C: per-seed violin distributions ────────────────────────────────
    split_keys = ["Main test set", "Hard subgroup", "External holdout"]
    vp_colors  = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
    vp_fills   = [PALETTE["blue_soft"], PALETTE["orange_soft"], PALETTE["green_soft"]]

    # Compute global y-range for consistent annotation placement
    all_vals = np.concatenate(list(per_seed.values()))
    y_lo = all_vals.min() - 0.5 * (all_vals.max() - all_vals.min()) * 0.12
    rng = np.random.default_rng(42)

    for xi, (key, col, fill) in enumerate(zip(split_keys, vp_colors, vp_fills)):
        data = per_seed[key]
        vp = ax_right.violinplot(data, positions=[xi], widths=0.62,
                                 showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(fill)
            body.set_edgecolor(col)
            body.set_linewidth(1.1)
            body.set_alpha(0.85)
        q25, med, q75 = np.percentile(data, [25, 50, 75])
        ax_right.plot([xi, xi], [q25, q75], color=col, lw=3.5, solid_capstyle="round", zorder=4)
        ax_right.scatter([xi], [med], s=55, color="white", edgecolor=col, linewidth=1.5, zorder=5)
        jx = rng.uniform(-0.18, 0.18, len(data))
        ax_right.scatter(xi + jx, data, s=9, color=col, alpha=0.30, edgecolor="none", zorder=3)

    ax_right.axhline(0, color=PALETTE["ink"], lw=0.9, linestyle="--", zorder=1)
    ax_right.set_xticks(range(len(split_keys)))
    ax_right.set_xticklabels([clean_label(k) for k in split_keys], fontsize=9.0)
    ax_right.set_ylabel("Per-seed MAE reduction (K)")
    ax_right.set_title("Distribution over 100 independent seeds", fontsize=11.0, pad=9)
    ax_right.grid(axis="y", linestyle=":", alpha=0.28)
    for sp in ("top", "right"):
        ax_right.spines[sp].set_visible(False)
    ax_right.text(0.02, 1.05, "C", transform=ax_right.transAxes, fontsize=12.5, fontweight="bold")

    # Stat annotations below each violin using axes-fraction y so they don't shift the ylim
    for xi, (key, col) in enumerate(zip(split_keys, vp_colors)):
        data = per_seed[key]
        ax_right.annotate(
            f"$\mu$={data.mean():.2f}\n$\sigma$={data.std():.2f} K",
            xy=(xi, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -42), textcoords="offset points",
            ha="center", va="top", fontsize=8.0, color=col,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor=col, linewidth=0.7, alpha=0.92),
        )

    fig.subplots_adjust(bottom=0.18)
    save(fig, "fig2_main_results")


def fig_cluster_reduction(diag_dir: Path) -> None:
    df = pd.read_csv(diag_dir / "cluster_results_table.csv")
    df = df.copy()
    lows, highs = [], []
    for t in df["95% CI"].tolist():
        lo, hi = parse_ci(t)
        lows.append(lo)
        highs.append(hi)
    df["ci_low"] = lows
    df["ci_high"] = highs
    df["label"] = df["Chemistry cluster"].map(clean_label)
    df = df.sort_values("MAE reduction (K)", ascending=True)

    y = np.arange(len(df))
    vals = df["MAE reduction (K)"].to_numpy(dtype=float)
    err_low = vals - df["ci_low"].to_numpy(dtype=float)
    err_high = df["ci_high"].to_numpy(dtype=float) - vals
    sample_counts = df["Sample count"].to_numpy(dtype=float)
    sizes = 55 + 0.75 * sample_counts
    colors = []
    for lo, hi, v in zip(df["ci_low"], df["ci_high"], vals):
        if hi <= 0:
            colors.append(PALETTE["red"])
        elif lo >= 0:
            colors.append(PALETTE["teal"])
        else:
            colors.append(PALETTE["orange"])

    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    fig.patch.set_facecolor(PALETTE["bg"])
    for yi in y:
        if yi % 2 == 0:
            ax.axhspan(yi - 0.45, yi + 0.45, color=PALETTE["blue_ll"], zorder=0)
    ax.hlines(y, df["ci_low"], df["ci_high"], color=PALETTE["ink"], linewidth=1.4, zorder=2)
    ax.scatter(vals, y, s=sizes, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    ax.axvline(0, color=PALETTE["ink"], linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    ax.set_xlabel("MAE reduction (K), baseline - final")
    ax.set_title("Cluster-wise transfer is heterogeneous; aggregate gain is positive", fontsize=11.7, pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for yi, v, count in zip(y, vals, sample_counts):
        ax.text(
            v + (0.06 if v >= 0 else -0.06),
            yi,
            f"{v:+.2f}  (n={int(count)})",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8.8,
            color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.82),
        )

    ax.set_facecolor("#FFFFFF")
    save(fig, "fig4_cluster_reduction")
def fig_overview() -> None:
    fig, ax = plt.subplots(figsize=(14.6, 6.8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "ink":       "#1A1A1A",
        "muted":     "#888888",
        "line":      "#CCCCCC",
        "soft_line": "#DDDDDD",
        "desc":      "#1B4F8A",   # descriptor branch → primary blue
        "desc_soft": "#D6E8F7",
        "graph":     "#2E75B6",   # graph branch → medium blue
        "graph_soft":"#EAF3FB",
        "ctx":       "#C05A16",   # polymer context → orange accent
        "ctx_soft":  "#FAE5D3",
        "mspce":     "#C05A16",   # MSCE → orange
        "mspce_soft":"#FAE5D3",
        "rcmf":      "#1B4F8A",   # RCMF → primary blue
        "rcmf_soft": "#D6E8F7",
        "masd":      "#1B4F8A",   # MASD → primary blue
        "masd_soft": "#D6E8F7",
        "plus":      "#1A6B3C",   # positive slot → dark green
        "minus":     "#8B1A1A",   # negative slot → dark red
        "train":     "#EEEEEE",
    }

    def box(x, y, w, h, *, fc="#FFFFFF", ec=None, lw=1.2, r=0.014, ls="-", z=1):
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec or colors["line"],
            linewidth=lw,
            linestyle=ls,
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def txt(x, y, s, *, size=10, weight=None, color=None, ha="center", va="center", style=None, z=5):
        ax.text(
            x,
            y,
            s,
            fontsize=size,
            fontweight=weight,
            color=color or colors["ink"],
            ha=ha,
            va=va,
            style=style,
            zorder=z,
        )

    def arr(x1, y1, x2, y2, *, color, lw=1.6, style="-|>", ms=12, ls="-", z=4):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle=style,
                lw=lw,
                color=color,
                mutation_scale=ms,
                linestyle=ls,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=z,
        )

    def descriptor_icon(x, y, w, h):
        for idx, hh in enumerate([0.26, 0.48, 0.72, 0.56]):
            bx = x + 0.08 * w + idx * 0.18 * w
            bw = 0.11 * w
            box(bx, y + 0.10 * h, bw, hh * h, fc="#F3D8AE", ec=colors["desc"], lw=1.0, r=0.004, z=4)

    def graph_icon(x, y, w, h):
        nodes = np.array(
            [
                [0.10, 0.48],
                [0.28, 0.72],
                [0.44, 0.54],
                [0.64, 0.72],
                [0.84, 0.50],
                [0.55, 0.24],
            ]
        )
        edges = [(0, 1), (1, 2), (2, 3), (2, 5), (3, 4), (0, 2), (4, 5)]
        for i, j in edges:
            ax.plot(
                [x + w * nodes[i, 0], x + w * nodes[j, 0]],
                [y + h * nodes[i, 1], y + h * nodes[j, 1]],
                color=colors["graph"],
                lw=1.4,
                zorder=4,
            )
        fills = ["#F3C69D", "#BFE0DB", "#D9E7B3", "#B7D2F0", "#F2D2B6", "#C7DFE7"]
        for idx, (nx, ny) in enumerate(nodes):
            ax.add_patch(
                patches.Circle(
                    (x + w * nx, y + h * ny),
                    radius=min(w, h) * 0.07,
                    facecolor=fills[idx % len(fills)],
                    edgecolor=colors["graph"],
                    lw=1.0,
                    zorder=5,
                )
            )

    def polymer_icon(x, y, w, h):
        xs = np.linspace(x + 0.08 * w, x + 0.92 * w, 10)
        ys = y + 0.48 * h + 0.14 * h * np.sin(np.linspace(0, 3.2 * np.pi, len(xs)))
        ax.plot(xs, ys, color=colors["ctx"], lw=1.6, zorder=4)
        fills = ["#D6E4F4", "#F4D9B6", "#D9E8C0", "#CBE6E1"]
        for idx, (xx, yy) in enumerate(zip(xs, ys)):
            ax.add_patch(
                patches.Circle(
                    (xx, yy),
                    radius=min(w, h) * 0.055,
                    facecolor=fills[idx % len(fills)],
                    edgecolor=colors["ctx"],
                    lw=0.9,
                    zorder=5,
                )
            )

    def module_shell(x, y, w, h, *, accent, fill, badge, title):
        box(x, y, w, h, fc=fill, ec=fill, lw=0.0, r=0.024, z=0)
        ax.plot([x + 0.018, x + w - 0.018], [y + h - 0.050, y + h - 0.050], color=accent, lw=2.1, zorder=2)
        box(x + 0.018, y + h - 0.090, 0.072, 0.043, fc="#FFFFFF", ec=accent, lw=1.3, r=0.010, z=3)
        txt(x + 0.054, y + h - 0.069, badge, size=9.8, weight="bold", color=accent)
        txt(x + 0.022, y + h - 0.112, title, size=10.4, weight="bold", color=accent, ha="left")

    def pill(x, y, w, h, label, *, fc="#FFFFFF", ec=None, color=None, size=9.3, weight="bold", alpha=1.0):
        patch = box(x, y, w, h, fc=fc, ec=ec, lw=1.2, r=0.012, z=3)
        patch.set_alpha(alpha)
        txt(x + w / 2, y + h / 2, label, size=size, weight=weight, color=color, z=4)

    def input_card(x, y, w, h, *, title, subtitle, accent, fill):
        box(x, y, w, h, fc=fill, ec=accent, lw=1.5, r=0.018, z=2)
        txt(x + 0.018, y + h - 0.042, title, size=11.0, weight="bold", ha="left")
        txt(x + 0.018, y + 0.036, subtitle, size=8.7, color=colors["muted"], ha="left")

    ax.plot([0.03, 0.97], [0.90, 0.90], color=colors["line"], lw=1.1, zorder=1)
    txt(0.12, 0.935, "Input views", size=13.0, weight="bold")
    txt(0.56, 0.935, "Core method pipeline", size=13.0, weight="bold")
    txt(0.92, 0.935, "Output", size=12.8, weight="bold")

    input_x, input_w, input_h = 0.04, 0.17, 0.18
    module_y, module_h = 0.22, 0.60
    m1 = (0.30, module_y, 0.18, module_h)
    m2 = (0.51, module_y, 0.19, module_h)
    m3 = (0.73, module_y, 0.15, module_h)

    input_card(input_x, 0.66, input_w, input_h, title="Descriptor view", subtitle="528-d chemical descriptors", accent=colors["desc"], fill=colors["desc_soft"])
    descriptor_icon(input_x + 0.020, 0.675, 0.10, 0.060)

    input_card(input_x, 0.43, input_w, input_h, title="Graph view", subtitle="AttentiveFP molecular graph", accent=colors["graph"], fill=colors["graph_soft"])
    graph_icon(input_x + 0.016, 0.448, 0.12, 0.075)

    input_card(input_x, 0.20, input_w, input_h, title="Polymer context", subtitle="980-d multiscale polymer context", accent=colors["ctx"], fill=colors["ctx_soft"])
    ctx_labels = ["chain n-grams", "segment window", "graph neighborhood", "polymer statistics"]
    for idx, label in enumerate(ctx_labels):
        pill(input_x + 0.016, 0.324 - idx * 0.028, 0.105, 0.022, label, fc="#FFFFFF", ec=colors["ctx"], color=colors["ctx"], size=6.7, weight=None)

    pill(0.225, 0.705, 0.075, 0.050, r"$y_d,\ u_d$", fc=colors["desc_soft"], ec=colors["desc"], color=colors["desc"], size=10.0)
    pill(0.225, 0.475, 0.075, 0.050, r"$y_g,\ u_g$", fc=colors["graph_soft"], ec=colors["graph"], color=colors["graph"], size=10.0)

    module_shell(*m1, accent=colors["mspce"], fill=colors["mspce_soft"], badge="1 MSCE", title="Hierarchical context selection")
    module_shell(*m2, accent=colors["rcmf"], fill=colors["rcmf_soft"], badge="2 RCMF", title="Reliability-conditioned fusion")
    module_shell(*m3, accent=colors["masd"], fill=colors["masd_soft"], badge="3 MASD", title="Bounded signed correction")

    m1_x, m1_y, m1_w, m1_h = m1
    scale_y = [m1_y + 0.39, m1_y + 0.31, m1_y + 0.23, m1_y + 0.15]
    for yy, label in zip(scale_y, ctx_labels):
        pill(m1_x + 0.020, yy, 0.088, 0.050, label, fc="#FFFFFF", ec=colors["ctx"], color=colors["ink"], size=8.9, weight=None)
    pill(m1_x + 0.123, m1_y + 0.245, 0.042, 0.105, "Top-k\nselector", fc="#FFF8EE", ec=colors["mspce"], color=colors["mspce"], size=9.1)
    txt(m1_x + 0.144, m1_y + 0.225, r"$k=3$", size=8.8, color=colors["muted"])
    pill(m1_x + 0.048, m1_y + 0.070, 0.105, 0.060, r"selected context $h_{ctx}$", fc="#FFFFFF", ec=colors["mspce"], color=colors["mspce"], size=9.2)
    txt(m1_x + m1_w / 2, m1_y + 0.028, "select active context scales", size=8.4, color=colors["muted"], style="italic")

    m2_x, m2_y, m2_w, m2_h = m2
    pill(m2_x + 0.020, m2_y + 0.405, 0.062, 0.050, r"$y_d,\ u_d$", fc=colors["desc_soft"], ec=colors["desc"], color=colors["desc"], size=9.5)
    pill(m2_x + 0.020, m2_y + 0.315, 0.062, 0.050, r"$y_g,\ u_g$", fc=colors["graph_soft"], ec=colors["graph"], color=colors["graph"], size=9.5)
    pill(m2_x + 0.095, m2_y + 0.405, 0.075, 0.052, r"$q = [y_d, y_g,$" + "\n" + r"$u_d, u_g, |y_d-y_g|]$", fc="#FFFFFF", ec=colors["rcmf"], color=colors["ink"], size=8.5)
    pill(m2_x + 0.102, m2_y + 0.295, 0.070, 0.115, "anchor +\nconstrained\nresidual fusion", fc="#FFFFFF", ec=colors["rcmf"], color=colors["rcmf"], size=9.4)
    pill(m2_x + 0.094, m2_y + 0.110, 0.078, 0.060, r"anchor prediction $y_{anchor}$", fc="#FFFFFF", ec=colors["rcmf"], color=colors["rcmf"], size=8.9)
    txt(m2_x + m2_w / 2, m2_y + 0.028, "reliability regulates cross-view interaction", size=8.4, color=colors["muted"], style="italic")

    m3_x, m3_y, m3_w, m3_h = m3
    pill(m3_x + 0.026, m3_y + 0.400, 0.096, 0.052, r"anchor prediction $y_{anchor}$", fc="#FFFFFF", ec=colors["masd"], color=colors["masd"], size=8.9)
    slot_xs = [m3_x + 0.018, m3_x + 0.052, m3_x + 0.086, m3_x + 0.120]
    slot_specs = [
        ("+", "#FBE7E5", colors["plus"], 1.0),
        ("+", "#FBE7E5", colors["plus"], 0.45),
        ("-", "#E7F0FB", colors["minus"], 1.0),
        ("-", "#E7F0FB", colors["minus"], 0.45),
    ]
    for xx, (label, fc, ec, alpha) in zip(slot_xs, slot_specs):
        patch = box(xx, m3_y + 0.280, 0.024, 0.100, fc=fc, ec=ec, lw=1.3, r=0.010, z=3)
        patch.set_alpha(alpha)
        txt(xx + 0.012, m3_y + 0.330, label, size=16, weight="bold", color=ec)
    pill(m3_x + 0.040, m3_y + 0.165, 0.085, 0.054, "sparse slot gate", fc="#FFFFFF", ec=colors["masd"], color=colors["masd"], size=9.4)
    pill(m3_x + 0.030, m3_y + 0.075, 0.105, 0.062, "bounded signed\ncorrection", fc="#FFFFFF", ec=colors["masd"], color=colors["masd"], size=9.4)
    txt(m3_x + m3_w / 2, m3_y + 0.028, "apply limited sparse correction", size=8.4, color=colors["muted"], style="italic")

    box(0.905, 0.355, 0.072, 0.225, fc="#FFFFFF", ec=colors["line"], lw=1.3, r=0.020, z=2)
    txt(0.941, 0.495, "Predicted\n$T_g$", size=12.4, weight="bold")
    txt(0.941, 0.407, "selective error\nreduction", size=8.9, color=colors["muted"])
    txt(0.941, 0.365, "largest gain on\nhard samples", size=8.4, color=colors["muted"], style="italic")

    box(0.06, 0.060, 0.78, 0.090, fc=colors["train"], ec=colors["soft_line"], lw=1.0, r=0.014, ls=(0, (4, 3)), z=1)
    txt(0.12, 0.107, "Geometry prior", size=10.4, weight="bold", ha="left")
    txt(0.12, 0.078, "train-time auxiliary only; not used at inference", size=8.6, color=colors["muted"], ha="left", style="italic")

    arr(input_x + input_w, 0.75, 0.225, 0.730, color=colors["desc"], lw=1.5)
    arr(input_x + input_w, 0.52, 0.225, 0.500, color=colors["graph"], lw=1.5)
    arr(input_x + input_w, 0.29, m1_x + 0.020, m1_y + 0.415, color=colors["ctx"], lw=1.6)
    for yy in [v + 0.025 for v in scale_y]:
        arr(m1_x + 0.108, yy, m1_x + 0.123, m1_y + 0.298, color=colors["mspce"], lw=1.0, ms=9)
    arr(m1_x + 0.144, m1_y + 0.245, m1_x + 0.100, m1_y + 0.130, color=colors["mspce"], lw=1.4)
    arr(0.300, 0.730, m2_x + 0.020, m2_y + 0.430, color=colors["desc"], lw=1.4)
    arr(0.300, 0.500, m2_x + 0.020, m2_y + 0.340, color=colors["graph"], lw=1.4)
    arr(m1_x + 0.153, m1_y + 0.100, m2_x + 0.102, m2_y + 0.350, color=colors["mspce"], lw=1.4)
    arr(m2_x + 0.132, m2_y + 0.405, m2_x + 0.137, m2_y + 0.410, color=colors["rcmf"], lw=1.5)
    arr(m2_x + 0.082, m2_y + 0.430, m2_x + 0.102, m2_y + 0.360, color=colors["desc"], lw=1.2, ms=9)
    arr(m2_x + 0.082, m2_y + 0.340, m2_x + 0.102, m2_y + 0.345, color=colors["graph"], lw=1.2, ms=9)
    arr(m2_x + 0.137, m2_y + 0.295, m2_x + 0.133, m2_y + 0.170, color=colors["rcmf"], lw=1.5)
    arr(m2_x + 0.172, m2_y + 0.140, m3_x + 0.026, m3_y + 0.425, color=colors["rcmf"], lw=1.5)
    arr(m3_x + 0.074, m3_y + 0.400, m3_x + 0.074, m3_y + 0.380, color=colors["masd"], lw=1.4)
    arr(slot_xs[0] + 0.012, m3_y + 0.280, m3_x + 0.082, m3_y + 0.219, color=colors["masd"], lw=1.1, ms=9)
    arr(slot_xs[2] + 0.012, m3_y + 0.280, m3_x + 0.082, m3_y + 0.219, color=colors["masd"], lw=1.1, ms=9)
    arr(m3_x + 0.082, m3_y + 0.165, m3_x + 0.082, m3_y + 0.137, color=colors["masd"], lw=1.4)
    arr(m3_x + 0.135, m3_y + 0.106, 0.905, 0.468, color=colors["masd"], lw=1.6)
    arr(0.53, 0.105, m2_x + 0.133, m2_y + 0.110, color=colors["muted"], lw=1.1, ls=(0, (4, 3)), ms=10)

    save(fig, "fig1_overview")


def fig_hardcase_evidence(diag_dir: Path) -> None:
    import json

    stats = json.loads((diag_dir / "stats.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(stats["per_seed_records"])
    split_specs = [
        ("Main test", "primary_mae_reduction_k", PALETTE["blue"]),
        ("Hard subgroup", "hard_mae_reduction_k", PALETTE["orange"]),
        ("External holdout", "external_mae_reduction_k", PALETTE["green"]),
    ]

    fig = plt.figure(figsize=(12.2, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.26)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor(PALETTE["bg"])

    positions = np.arange(1, len(split_specs) + 1)
    data = [df[col].to_numpy(dtype=float) for _, col, _ in split_specs]
    bp = ax_left.boxplot(
        data,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=PALETTE["ink"], linewidth=1.5),
        whiskerprops=dict(color=PALETTE["gray"], linewidth=1.2),
        capprops=dict(color=PALETTE["gray"], linewidth=1.2),
    )
    for patch, (_, _, color) in zip(bp["boxes"], split_specs):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)
    rng = np.random.default_rng(7)
    for idx, ((label, col, color), vals) in enumerate(zip(split_specs, data), start=1):
        x = idx + rng.normal(0.0, 0.045, size=len(vals))
        ax_left.scatter(x, vals, s=20, color=color, alpha=0.32, edgecolor="none", zorder=3)
        pos = int(np.sum(vals > 0))
        ax_left.text(idx, np.max(vals) + (0.6 if idx == 2 else 0.22), f"{pos}/100 positive", ha="center", va="bottom", fontsize=8.7, color=PALETTE["ink"])

    ax_left.axhline(0, color=PALETTE["ink"], linestyle="--", linewidth=1.0, zorder=1)
    ax_left.set_xticks(positions)
    ax_left.set_xticklabels([label for label, _, _ in split_specs])
    ax_left.set_ylabel("Run-wise MAE reduction (K)")
    ax_left.set_title("A. Seed-wise reduction distributions", fontsize=11.6, pad=10)
    ax_left.grid(axis="y", linestyle=":", alpha=0.28)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    base = df["baseline_hard_mae_k"].to_numpy(dtype=float)
    final = df["final_hard_mae_k"].to_numpy(dtype=float)
    for b, f in zip(base, final):
        ax_right.plot([0, 1], [b, f], color=PALETTE["line"], lw=0.9, alpha=0.45, zorder=1)
    ax_right.scatter(np.zeros_like(base), base, s=18, color=PALETTE["gray"], alpha=0.65, edgecolor="white", linewidth=0.4, zorder=2)
    ax_right.scatter(np.ones_like(final), final, s=20, color=PALETTE["orange"], alpha=0.72, edgecolor="white", linewidth=0.4, zorder=3)
    ax_right.scatter([0, 1], [base.mean(), final.mean()], s=[120, 140], color=[PALETTE["gray"], PALETTE["orange"]], edgecolor=PALETTE["ink"], linewidth=0.8, zorder=5)
    ax_right.text(-0.15, base.mean(), f"mean {base.mean():.2f}", ha="right", va="center", fontsize=8.8, color=PALETTE["gray"])
    ax_right.text(1.15, final.mean(), f"mean {final.mean():.2f}", ha="left", va="center", fontsize=8.8, color=PALETTE["ink"])
    ax_right.text(0.50, max(base.max(), final.max()) + 0.85, "Mean hard-subgroup reduction: 4.22 K", ha="center", fontsize=9.2, fontweight="bold", color=PALETTE["ink"])
    ax_right.text(0.50, max(base.max(), final.max()) + 0.25, "largest paired shift on the hard subgroup", ha="center", fontsize=8.8, color=PALETTE["gray"], style="italic")

    ax_right.set_xlim(-0.42, 1.42)
    ax_right.set_xticks([0, 1])
    ax_right.set_xticklabels(["Baseline", "MSCE-RCMF-MASD"])
    ax_right.set_ylabel("Hard-subgroup MAE (K)")
    ax_right.set_title("B. Paired hard-subgroup shift across 100 runs", fontsize=11.6, pad=10)
    ax_right.grid(axis="y", linestyle=":", alpha=0.28)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    save(fig, "fig3_hardcase_evidence")


def _legacy_fig_loss_sketch_unused() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, *, fc="#FFFFFF", ec=None, lw=1.2, r=0.015):
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec or PALETTE["ink"],
            linewidth=lw,
        )
        ax.add_patch(patch)
        return patch

    def txt(x, y, s, *, size=10, weight=None, color=None, ha="center", va="center", style=None):
        ax.text(x, y, s, fontsize=size, fontweight=weight, color=color or PALETTE["ink"], ha=ha, va=va, style=style)

    def arr(x1, y1, x2, y2, *, color, lw=1.4, ls="-", ms=11):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, linestyle=ls, mutation_scale=ms, shrinkA=0, shrinkB=0),
        )

    txt(0.20, 0.93, "A. refinement ladder", size=12.0, weight="bold")
    txt(0.69, 0.93, "B. grouped training controls", size=12.0, weight="bold")

    ladder = [
        (0.05, 0.74, 0.25, 0.10, "#EEF3F6", "Simple Concat baseline", "branch supervision"),
        (0.05, 0.58, 0.25, 0.10, "#FCF1E8", "MSCE repair anchor", "context-preserving repair"),
        (0.05, 0.42, 0.25, 0.10, "#E6F2F4", "RCMF fusion anchor", "reliability-conditioned fusion"),
        (0.05, 0.26, 0.25, 0.10, "#EFEAF7", "MASD final correction", "signed sparse bounded correction"),
    ]
    for x, y, w, h, fc, title, subtitle in ladder:
        box(x, y, w, h, fc=fc, ec=PALETTE["ink"], lw=1.1)
        txt(x + 0.02, y + 0.063, title, size=10.4, weight="bold", ha="left")
        txt(x + 0.02, y + 0.032, subtitle, size=8.7, ha="left", color=PALETTE["gray"])
    for (_, y1, _, _, _, _, _), (_, y2, _, _, _, _, _) in zip(ladder[:-1], ladder[1:]):
        arr(0.175, y1, 0.175, y2 + 0.10, color=PALETTE["ink"])

    family_specs = [
        (0.37, 0.67, 0.24, 0.16, "#EEF3F6", "Supervision", ["branch losses", "main prediction", "anchor fit"]),
        (0.64, 0.67, 0.28, 0.16, "#FCF1E8", "Anchor consistency", ["repair preservation", "anchor margin", "context path retention"]),
        (0.37, 0.45, 0.24, 0.16, "#E6F2F4", "Reliability / gate control", ["bounded gate behavior", "uncertainty-aware regulation", "fusion safety"]),
        (0.64, 0.45, 0.28, 0.16, "#EFEAF7", "Signed decomposition", ["sign consistency", "sparse slot allocation", "slot diversity"]),
        (0.50, 0.23, 0.28, 0.15, "#F4F1EB", "Hard-case stabilization", ["hard subgroup stability", "weak-cluster guardrails", "risk-aware correction limits"]),
    ]
    for x, y, w, h, fc, title, bullets in family_specs:
        box(x, y, w, h, fc=fc, ec=PALETTE["ink"], lw=1.0)
        txt(x + 0.015, y + h - 0.038, title, size=10.0, weight="bold", ha="left", va="top")
        for idx, bullet in enumerate(bullets):
            txt(x + 0.020, y + h - 0.075 - idx * 0.030, f"• {bullet}", size=8.6, ha="left", va="top", color=PALETTE["gray"])

    # stage strip
    box(0.37, 0.07, 0.55, 0.09, fc="#FFFFFF", ec=PALETTE["ink"], lw=1.0)
    for xpos in [0.553, 0.736]:
        ax.plot([xpos, xpos], [0.07, 0.16], color=PALETTE["line"], lw=1.0)
    txt(0.46, 0.132, "Stage A", size=9.8, weight="bold")
    txt(0.645, 0.132, "Stage B", size=9.8, weight="bold")
    txt(0.83, 0.132, "Stage C", size=9.8, weight="bold")
    txt(0.46, 0.096, "supervision\nanchor consistency", size=8.0, color=PALETTE["gray"])
    txt(0.645, 0.096, " + gate control", size=8.0, color=PALETTE["gray"])
    txt(0.83, 0.096, " + signed decomposition\n + hard-case stabilization", size=8.0, color=PALETTE["gray"])

    # control arrows
    arr(0.30, 0.63, 0.37, 0.75, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=10)
    arr(0.30, 0.47, 0.37, 0.53, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=10)
    arr(0.30, 0.31, 0.37, 0.30, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=10)
    arr(0.61, 0.45, 0.61, 0.16, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=10)
    arr(0.78, 0.45, 0.83, 0.16, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=10)

    txt(0.50, 0.015, "The appendix figure groups loss terms by function and training stage rather than listing every scalar coefficient separately.", size=8.6, color=PALETTE["gray"], style="italic")

    save(fig, "figA1_loss_sketch")


def fig_loss_sketch() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.9))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, *, fc="#FFFFFF", ec=None, lw=1.2, r=0.015):
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec or PALETTE["ink"],
            linewidth=lw,
        )
        ax.add_patch(patch)
        return patch

    def txt(x, y, s, *, size=10, weight=None, color=None, ha="center", va="center", style=None):
        ax.text(x, y, s, fontsize=size, fontweight=weight, color=color or PALETTE["ink"], ha=ha, va=va, style=style)

    def chip(x, y, w, h, label, *, fc="#FFFFFF", ec=None, lw=0.9, color=None, size=8.1):
        patch = box(x, y, w, h, fc=fc, ec=ec or PALETTE["line"], lw=lw, r=0.012)
        txt(x + w / 2, y + h / 2, label, size=size, weight="bold", color=color or PALETTE["gray"])
        return patch

    def arr(x1, y1, x2, y2, *, color, lw=1.4, ls="-", ms=11, connectionstyle=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=lw,
                color=color,
                linestyle=ls,
                mutation_scale=ms,
                shrinkA=0,
                shrinkB=0,
                connectionstyle=connectionstyle,
            ),
        )

    txt(0.20, 0.93, "A. refinement ladder", size=12.4, weight="bold")
    txt(0.72, 0.93, "B. grouped training controls", size=12.4, weight="bold")
    ax.plot([0.40, 0.40], [0.14, 0.88], color=PALETTE["line"], lw=1.0)

    ladder = [
        (0.05, 0.73, 0.25, 0.105, PALETTE["blue_soft"],   PALETTE["blue"],   "Simple Concat baseline", "branch supervision", "baseline\nprediction"),
        (0.05, 0.57, 0.25, 0.105, PALETTE["orange_soft"], PALETTE["orange"], "MSCE repair anchor", "context-preserving repair", "repaired\nanchor"),
        (0.05, 0.41, 0.25, 0.105, PALETTE["blue_ll"],     PALETTE["slate"],  "RCMF fusion anchor", "reliability-conditioned fusion", "fusion\nanchor"),
        (0.05, 0.25, 0.25, 0.105, PALETTE["blue_soft"],   PALETTE["blue"],   "MASD final correction", "signed sparse bounded correction", "final\nprediction"),
    ]
    for idx, (x, y, w, h, fc, accent, title, subtitle, output_label) in enumerate(ladder, start=1):
        box(x, y, w, h, fc=fc, ec=PALETTE["ink"], lw=1.1)
        badge = patches.Circle((x + 0.030, y + 0.064), 0.018, facecolor=accent, edgecolor=PALETTE["ink"], linewidth=0.9)
        ax.add_patch(badge)
        txt(x + 0.030, y + 0.064, str(idx), size=8.6, weight="bold", color="#FFFFFF")
        txt(x + 0.055, y + 0.068, title, size=10.4, weight="bold", ha="left")
        txt(x + 0.055, y + 0.035, subtitle, size=8.7, ha="left", color=PALETTE["gray"])
        chip(x + w + 0.016, y + 0.032, 0.095, 0.044, output_label, fc="#FFFFFF", ec=PALETTE["line"], size=7.2)
    for (_, y1, _, _, _, _, _, _, _), (_, y2, _, _, _, _, _, _, _) in zip(ladder[:-1], ladder[1:]):
        arr(0.175, y1, 0.175, y2 + 0.10, color=PALETTE["ink"])

    family_specs = [
        (0.44, 0.65, 0.22, 0.15, PALETTE["blue_soft"],   "Stage A", "Supervision", ["branch supervision", "main prediction loss", "anchor fit"]),
        (0.70, 0.65, 0.23, 0.15, PALETTE["orange_soft"], "Stage A", "Anchor consistency", ["anchor preservation", "repair consistency", "context path retention"]),
        (0.44, 0.43, 0.22, 0.15, PALETTE["blue_ll"],     "Stage B", "Reliability / gate control", ["bounded gate behavior", "uncertainty-aware regulation", "fusion safety"]),
        (0.70, 0.43, 0.23, 0.15, PALETTE["blue_soft"],   "Stage C", "Signed decomposition", ["sign consistency", "sparse slot allocation", "slot diversity"]),
        (0.56, 0.22, 0.25, 0.14, PALETTE["slate_soft"],  "Stage C", "Hard-case stabilization", ["hard subgroup stability", "weak-cluster guardrails", "risk-aware correction limits"]),
    ]
    for x, y, w, h, fc, stage, title, bullets in family_specs:
        box(x, y, w, h, fc=fc, ec=PALETTE["ink"], lw=1.0)
        chip(x + w - 0.085, y + h - 0.040, 0.070, 0.028, stage, fc="#FFFFFF", ec=PALETTE["line"], size=7.0)
        txt(x + 0.015, y + h - 0.042, title, size=10.0, weight="bold", ha="left", va="top")
        for idx, bullet in enumerate(bullets):
            txt(x + 0.020, y + h - 0.086 - idx * 0.031, f"• {bullet}", size=8.5, ha="left", va="top", color=PALETTE["gray"])

    box(0.44, 0.07, 0.49, 0.09, fc="#FFFFFF", ec=PALETTE["ink"], lw=1.0)
    for xpos in [0.603, 0.766]:
        ax.plot([xpos, xpos], [0.07, 0.16], color=PALETTE["line"], lw=1.0)
    txt(0.522, 0.132, "Stage A", size=9.6, weight="bold")
    txt(0.685, 0.132, "Stage B", size=9.6, weight="bold")
    txt(0.848, 0.132, "Stage C", size=9.6, weight="bold")
    txt(0.522, 0.096, "anchor-consistent\nsupervision", size=7.9, color=PALETTE["gray"])
    txt(0.685, 0.096, "+ reliability-aware\nfusion control", size=7.9, color=PALETTE["gray"])
    txt(0.848, 0.096, "+ signed correction\n+ hard-case guardrails", size=7.9, color=PALETTE["gray"])

    arr(0.30, 0.78, 0.44, 0.73, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=9, connectionstyle="arc3,rad=-0.18")
    arr(0.30, 0.62, 0.70, 0.79, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=9, connectionstyle="arc3,rad=0.18")
    arr(0.30, 0.46, 0.44, 0.50, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=9, connectionstyle="arc3,rad=-0.10")
    arr(0.30, 0.30, 0.70, 0.50, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=9, connectionstyle="arc3,rad=0.12")
    arr(0.815, 0.43, 0.685, 0.36, color=PALETTE["gray"], ls=(0, (4, 3)), lw=1.0, ms=9)

    txt(0.50, 0.015, "The appendix figure groups loss terms by function and training stage rather than listing every scalar coefficient separately.", size=8.6, color=PALETTE["gray"], style="italic")

    save(fig, "figA1_loss_sketch")


def fig_msce_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(15.4, 8.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "ink":         "#1A1A1A",
        "muted":       "#888888",
        "line":        "#CCCCCC",
        "soft":        "#EAF3FB",
        "input":       "#1B4F8A",   # primary blue
        "input_soft":  "#D6E8F7",
        "chain":       "#1B4F8A",   # S1 chain → primary blue
        "chain_soft":  "#D6E8F7",
        "segment":     "#C05A16",   # S2 segment → orange
        "segment_soft":"#FAE5D3",
        "graph":       "#2E75B6",   # S3 graph → medium blue
        "graph_soft":  "#EAF3FB",
        "stats":       "#5B7FA6",   # S4 statistics → slate
        "stats_soft":  "#EAF3FB",
        "gate":        "#C05A16",   # gate → orange
        "gate_soft":   "#FAE5D3",
        "fuse":        "#1A6B3C",   # fusion → green (positive output)
        "fuse_soft":   "#D5EDDF",
        "warn":        "#C05A16",   # warning → orange
        "warn_soft":   "#FAE5D3",
    }

    def box(x, y, w, h, *, fc="#FFFFFF", ec=None, lw=1.2, r=0.014, ls="-", z=1):
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec or colors["line"],
            linewidth=lw,
            linestyle=ls,
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def txt(x, y, s, *, size=10.0, weight=None, color=None, ha="center", va="center", style=None, z=5):
        ax.text(
            x,
            y,
            s,
            fontsize=size,
            fontweight=weight,
            color=color or colors["ink"],
            ha=ha,
            va=va,
            style=style,
            zorder=z,
        )

    def arr(x1, y1, x2, y2, *, color, lw=1.4, style="-|>", ms=12, ls="-", z=4, rad=0.0):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle=style,
                lw=lw,
                color=color,
                mutation_scale=ms,
                linestyle=ls,
                shrinkA=0,
                shrinkB=0,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=z,
        )

    def chip(x, y, w, h, label, *, fc="#FFFFFF", ec=None, color=None, size=8.4, weight="bold"):
        box(x, y, w, h, fc=fc, ec=ec, lw=1.0, r=0.010, z=3)
        txt(x + w / 2, y + h / 2, label, size=size, weight=weight, color=color)

    def token_strip(x, y, labels, *, active_start=None, active_len=0, w=0.027, h=0.038, gap=0.004, size=8.4):
        for idx, label in enumerate(labels):
            active = active_start is not None and active_start <= idx < active_start + active_len
            fc = colors["segment_soft"] if active else "#FFFFFF"
            ec = colors["segment"] if active else colors["line"]
            box(x + idx * (w + gap), y, w, h, fc=fc, ec=ec, lw=1.0, r=0.006, z=3)
            txt(x + idx * (w + gap) + w / 2, y + h / 2, label, size=size, color=colors["ink"])
        if active_start is not None:
            x0 = x + active_start * (w + gap)
            x1 = x0 + active_len * w + (active_len - 1) * gap
            ax.plot([x0, x1], [y - 0.008, y - 0.008], color=colors["segment"], lw=1.5, zorder=4)

    def polymer_repeat_icon(x, y, w, h):
        xs = np.linspace(x + 0.10 * w, x + 0.88 * w, 9)
        ys = y + 0.55 * h + 0.12 * h * np.sin(np.linspace(0, 2.8 * np.pi, len(xs)))
        ax.plot(xs, ys, color=colors["input"], lw=1.8, zorder=4)
        fills = ["#D6E8F7", "#FAE5D3", "#D5EDDF", "#EAF3FB"]
        for idx, (xx, yy) in enumerate(zip(xs, ys)):
            ax.add_patch(
                patches.Circle(
                    (xx, yy),
                    radius=min(w, h) * 0.040,
                    facecolor=fills[idx % len(fills)],
                    edgecolor=colors["input"],
                    lw=1.0,
                    zorder=5,
                )
            )
        ax.plot([x + 0.05 * w, x + 0.08 * w], [y + 0.55 * h, y + 0.55 * h], color=colors["input"], lw=1.4, zorder=4)
        ax.plot([x + 0.90 * w, x + 0.95 * w], [y + 0.55 * h, y + 0.55 * h], color=colors["input"], lw=1.4, zorder=4)
        txt(x + 0.02 * w, y + 0.55 * h, "[*]", size=8.2, color=colors["muted"])
        txt(x + 0.98 * w, y + 0.55 * h, "[*]", size=8.2, color=colors["muted"])
        txt(x + 0.94 * w, y + 0.78 * h, r"$n$", size=9.0, color=colors["muted"])

    def graph_neighborhood_icon(x, y, w, h):
        nodes = np.array(
            [
                [0.10, 0.50],
                [0.26, 0.73],
                [0.42, 0.54],
                [0.58, 0.74],
                [0.82, 0.56],
                [0.62, 0.28],
                [0.30, 0.26],
            ]
        )
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (0, 6), (6, 2), (5, 4)]
        for i, j in edges:
            ax.plot(
                [x + w * nodes[i, 0], x + w * nodes[j, 0]],
                [y + h * nodes[i, 1], y + h * nodes[j, 1]],
                color=colors["graph"],
                lw=1.3,
                zorder=4,
            )
        for idx, (nx, ny) in enumerate(nodes):
            face = "#EAF3FB" if idx in {2, 5} else "#FFFFFF"
            radius = 0.085 if idx == 2 else 0.060
            ax.add_patch(
                patches.Circle(
                    (x + w * nx, y + h * ny),
                    radius=min(w, h) * radius,
                    facecolor=face,
                    edgecolor=colors["graph"],
                    lw=1.0,
                    zorder=5,
                )
            )
        ax.add_patch(
            patches.Circle(
                (x + w * nodes[2, 0], y + h * nodes[2, 1]),
                radius=min(w, h) * 0.18,
                facecolor="none",
                edgecolor=colors["graph"],
                lw=1.0,
                linestyle=(0, (3, 2)),
                zorder=3,
            )
        )
        ax.add_patch(
            patches.Circle(
                (x + w * nodes[2, 0], y + h * nodes[2, 1]),
                radius=min(w, h) * 0.30,
                facecolor="none",
                edgecolor=colors["graph"],
                lw=1.0,
                linestyle=(0, (5, 3)),
                zorder=3,
            )
        )

    def stat_glyph(x, y, w, h):
        vals = [0.38, 0.58, 0.42, 0.70]
        fills = [colors["stats"], "#5B7FA6", "#8AAEC8", "#B3CEDF"]
        for idx, (val, fill) in enumerate(zip(vals, fills)):
            bx = x + 0.10 * w + idx * 0.18 * w
            bw = 0.10 * w
            box(bx, y + 0.12 * h, bw, val * h, fc=fill, ec=fill, lw=0.0, r=0.004, z=4)
        txt(x + 0.78 * w, y + 0.63 * h, "20-d", size=9.2, weight="bold", color=colors["stats"])

    def scale_card(x, y, w, h, *, accent, fill, badge, title, subtitle, bullets):
        box(x, y, w, h, fc=fill, ec=accent, lw=1.3, r=0.018, z=2)
        chip(x + 0.016, y + h - 0.045, 0.050, 0.030, badge, fc="#FFFFFF", ec=accent, color=accent, size=8.0)
        txt(x + 0.078, y + h - 0.030, title, ha="left", size=10.3, weight="bold")
        txt(x + 0.078, y + h - 0.060, subtitle, ha="left", size=8.2, color=colors["muted"])
        for idx, bullet in enumerate(bullets):
            txt(x + 0.020, y + h - 0.098 - idx * 0.029, f"• {bullet}", ha="left", va="top", size=8.2, color=colors["ink"])

    def lane_header(x, w, title):
        txt(x + w / 2, 0.915, title, size=12.4, weight="bold")
        ax.plot([x, x + w], [0.897, 0.897], color=colors["line"], lw=1.0)

    txt(0.03, 0.965, "MSCE: Multiscale Context Encoding", size=15.6, weight="bold", ha="left")
    txt(
        0.03,
        0.935,
        "Only the first innovation is shown here. MSCE takes one polymer 2D representation at inference time and decomposes it into multiple semantic context scales.",
        size=9.4,
        color=colors["muted"],
        ha="left",
    )

    left_x, left_y, left_w, left_h = 0.035, 0.090, 0.33, 0.78
    mid_x, mid_y, mid_w, mid_h = 0.385, 0.090, 0.26, 0.78
    right_x, right_y, right_w, right_h = 0.665, 0.090, 0.30, 0.78

    lane_header(left_x, left_w, "Inference-time Input State")
    lane_header(mid_x, mid_w, "Four Semantic Context Scales")
    lane_header(right_x, right_w, "MSCE Selection and Fusion")

    box(left_x, left_y, left_w, left_h, fc=colors["input_soft"], ec=colors["input"], lw=1.5, r=0.020)
    txt(left_x + 0.018, left_y + left_h - 0.038, "What the model actually sees", ha="left", size=10.9, weight="bold")
    txt(left_x + 0.018, left_y + left_h - 0.060, "one polymer 2D input -> ordered tokens + 2D graph", ha="left", size=8.5, color=colors["muted"])

    box(left_x + 0.018, left_y + 0.585, left_w - 0.036, 0.160, fc="#FFFFFF", ec=colors["line"], lw=1.0, r=0.016)
    txt(left_x + 0.034, left_y + 0.695, "Single polymer 2D input", ha="left", size=10.0, weight="bold")
    txt(left_x + 0.034, left_y + 0.670, "repeat-unit string + the same 2D graph", ha="left", size=8.0, color=colors["muted"])
    polymer_repeat_icon(left_x + 0.032, left_y + 0.605, left_w - 0.072, 0.090)
    chip(left_x + 0.034, left_y + 0.600, 0.094, 0.028, "2D only", fc=colors["warn_soft"], ec=colors["warn"], color=colors["warn"], size=8.0)
    chip(left_x + 0.140, left_y + 0.600, 0.120, 0.028, "no 3D conformer", fc="#FFF7ED", ec="#D9A14A", color="#9C6F19", size=7.8, weight="bold")

    box(left_x + 0.018, left_y + 0.340, left_w - 0.036, 0.220, fc="#FFFFFF", ec=colors["line"], lw=1.0, r=0.016)
    txt(left_x + 0.034, left_y + 0.525, "Chain and segment states come from the token stream", ha="left", size=9.8, weight="bold")
    txt(left_x + 0.034, left_y + 0.500, "draw the segment as a highlighted local token window", ha="left", size=8.1, color=colors["muted"])
    txt(left_x + 0.034, left_y + 0.462, "Chain state", ha="left", size=8.8, weight="bold", color=colors["chain"])
    token_strip(left_x + 0.034, left_y + 0.407, ["C", "O", "C", "(", "=", "O", ")", "c", "c"], w=0.030, h=0.040, gap=0.004, size=8.8)
    txt(left_x + 0.034, left_y + 0.378, "ordered polymer token stream", ha="left", size=7.9, color=colors["muted"])
    txt(left_x + 0.034, left_y + 0.340, "Segment state", ha="left", size=8.8, weight="bold", color=colors["segment"])
    token_strip(left_x + 0.034, left_y + 0.286, ["C", "O", "C", "(", "=", "O", ")", "c", "c"], active_start=2, active_len=5, w=0.030, h=0.040, gap=0.004, size=8.8)
    txt(left_x + 0.034, left_y + 0.282, "highlighted token window (w = 5), not a 3D fragment", ha="left", size=7.5, color=colors["segment"], style="italic")

    box(left_x + 0.018, left_y + 0.060, left_w - 0.036, 0.215, fc="#FFFFFF", ec=colors["line"], lw=1.0, r=0.016)
    txt(left_x + 0.034, left_y + 0.245, "Graph and interpretable states use the same 2D input", ha="left", size=9.9, weight="bold")
    txt(left_x + 0.034, left_y + 0.221, "neighborhood motifs are topological; statistics are token/graph summaries", ha="left", size=7.9, color=colors["muted"])
    txt(left_x + 0.034, left_y + 0.188, "2D graph neighborhoods", ha="left", size=8.8, weight="bold", color=colors["graph"])
    graph_neighborhood_icon(left_x + 0.038, left_y + 0.102, 0.145, 0.090)
    txt(left_x + 0.190, left_y + 0.145, "radius-1 / radius-2\natom-bond motifs", ha="left", size=8.1, color=colors["graph"])
    txt(left_x + 0.034, left_y + 0.078, "Interpretable statistics", ha="left", size=8.8, weight="bold", color=colors["stats"])
    stat_glyph(left_x + 0.175, left_y + 0.058, 0.110, 0.078)
    txt(left_x + 0.296, left_y + 0.078, "ring / aromatic /\nhetero / size ratios", ha="right", size=7.6, color=colors["stats"])

    box(mid_x, mid_y, mid_w, mid_h, fc=colors["soft"], ec=colors["line"], lw=1.2, r=0.020)
    txt(mid_x + 0.018, mid_y + mid_h - 0.038, "MSCE multiscale context construction", ha="left", size=11.0, weight="bold")
    txt(mid_x + 0.018, mid_y + mid_h - 0.068, "all four scales are derived from the same polymer 2D input", ha="left", size=8.4, color=colors["muted"])

    scale_card(
        mid_x + 0.014, 0.646, mid_w - 0.028, 0.125,
        accent=colors["chain"], fill=colors["chain_soft"], badge="S1",
        title="Chain n-gram context",
        subtitle="hashed 2/3/4-gram counts -> 576-d",
        bullets=["from the full ordered token stream", "captures repeat-unit order regularities"],
    )
    token_strip(mid_x + 0.032, 0.666, ["C", "O", "C", "=", "O", "c"], active_start=0, active_len=3, w=0.026, h=0.033, gap=0.004, size=8.0)

    scale_card(
        mid_x + 0.014, 0.497, mid_w - 0.028, 0.125,
        accent=colors["segment"], fill=colors["segment_soft"], badge="S2",
        title="Segment-window context",
        subtitle="sliding token windows (w = 5) -> 128-d",
        bullets=["from local token windows", "this is the correct segment modality to draw"],
    )
    token_strip(mid_x + 0.032, 0.517, ["C", "O", "C", "(", "=", "O", ")", "c"], active_start=1, active_len=5, w=0.023, h=0.033, gap=0.004, size=8.0)

    scale_card(
        mid_x + 0.014, 0.348, mid_w - 0.028, 0.125,
        accent=colors["graph"], fill=colors["graph_soft"], badge="S3",
        title="2D graph context",
        subtitle="radius-1 + radius-2 motifs -> 256-d",
        bullets=["from the same 2D repeat-unit graph", "captures atom-bond neighborhoods, not 3D geometry"],
    )
    graph_neighborhood_icon(mid_x + 0.033, 0.365, 0.100, 0.060)

    scale_card(
        mid_x + 0.014, 0.199, mid_w - 0.028, 0.125,
        accent=colors["stats"], fill=colors["stats_soft"], badge="S4",
        title="Interpretable statistics",
        subtitle="token / graph statistics -> 20-d",
        bullets=["counts and ratios from tokens and 2D graph", "ring, aromatic, hetero, HBA/HBD and size statistics"],
    )
    stat_glyph(mid_x + 0.030, 0.216, 0.110, 0.062)

    box(right_x, right_y, right_w, right_h, fc="#FFFFFF", ec=colors["line"], lw=1.2, r=0.020)
    txt(right_x + 0.018, right_y + right_h - 0.038, "MSCE selection + fusion path", ha="left", size=11.0, weight="bold")
    txt(right_x + 0.018, right_y + right_h - 0.068, "per-scale encoding -> sparse top-k gate -> fused context embedding", ha="left", size=8.4, color=colors["muted"])

    box(right_x + 0.020, 0.684, right_w - 0.040, 0.088, fc=colors["gate_soft"], ec=colors["gate"], lw=1.2, r=0.016)
    txt(right_x + 0.040, 0.742, "1. Scale encoders", ha="left", size=10.4, weight="bold")
    txt(right_x + 0.040, 0.713, r"each scale $c_s \rightarrow e_s \in \mathbb{R}^{128}$", ha="left", size=9.1)
    txt(right_x + right_w - 0.040, 0.713, "4 parallel scale embeddings", ha="right", size=8.0, color=colors["muted"])

    box(right_x + 0.020, 0.568, right_w - 0.040, 0.090, fc=colors["gate_soft"], ec=colors["gate"], lw=1.2, r=0.016)
    txt(right_x + 0.040, 0.627, "2. Dense scale scoring", ha="left", size=10.4, weight="bold")
    txt(right_x + 0.040, 0.598, r"$\tilde{w} = \mathrm{softmax}(\mathrm{score}(e_s))$", ha="left", size=9.1)
    txt(right_x + right_w - 0.040, 0.598, "relative importance over 4 scales", ha="right", size=8.0, color=colors["muted"])

    box(right_x + 0.020, 0.418, right_w - 0.040, 0.122, fc="#FFFFFF", ec=colors["gate"], lw=1.3, r=0.018)
    txt(right_x + 0.040, 0.514, "3. Sparse top-k gate", ha="left", size=10.5, weight="bold", color=colors["gate"])
    txt(right_x + 0.040, 0.486, r"keep top-$k$ scales with $k = 3$; one scale is suppressed per sample", ha="left", size=8.4, color=colors["muted"])
    bars = [0.20, 0.84, 0.62, 0.47]
    bx0 = right_x + 0.060
    for idx, height in enumerate(bars):
        xx = bx0 + idx * 0.052
        active = idx in {1, 2, 3}
        fc = colors["gate"] if active else "#E7E2DB"
        ec = colors["gate"] if active else colors["line"]
        alpha = 1.0 if active else 0.55
        patch = box(xx, 0.522, 0.028, 0.080 * height + 0.010, fc=fc, ec=ec, lw=0.0 if active else 1.0, r=0.004, z=4)
        patch.set_alpha(alpha)
        txt(xx + 0.014, 0.510, f"{idx+1}", size=8.0, color=colors["muted"])
    txt(right_x + 0.190, 0.458, r"active scales receive sparse weights $\alpha_s$", ha="left", size=8.3, color=colors["gate"])

    box(right_x + 0.020, 0.210, right_w - 0.040, 0.170, fc=colors["fuse_soft"], ec=colors["fuse"], lw=1.3, r=0.018)
    txt(right_x + 0.040, 0.353, "4. Context fusion", ha="left", size=10.5, weight="bold", color=colors["fuse"])
    txt(right_x + 0.040, 0.323, r"selected-scale embedding: $\sum_s \alpha_s e_s$", ha="left", size=9.0)
    txt(right_x + 0.040, 0.294, r"pooled active summary: mean(active $e_s$)", ha="left", size=9.0)
    txt(right_x + 0.040, 0.265, r"weak inactive residual: $0.20 \times$ inactive embedding", ha="left", size=9.0)
    txt(right_x + 0.040, 0.233, r"MLP fuse $\rightarrow h_{ctx}$", ha="left", size=9.4, weight="bold", color=colors["fuse"])

    box(right_x + 0.020, 0.096, right_w - 0.040, 0.085, fc="#FFFFFF", ec=colors["fuse"], lw=1.3, r=0.018)
    txt(right_x + 0.040, 0.151, "5. Output of the first innovation", ha="left", size=9.9, weight="bold")
    chip(right_x + 0.040, 0.108, 0.078, 0.032, r"$h_{ctx}$", fc=colors["fuse_soft"], ec=colors["fuse"], color=colors["fuse"], size=10.4)
    chip(right_x + 0.126, 0.108, 0.056, 0.032, "chain", fc=colors["chain_soft"], ec=colors["chain"], color=colors["chain"], size=8.1)
    chip(right_x + 0.188, 0.108, 0.068, 0.032, "segment", fc=colors["segment_soft"], ec=colors["segment"], color=colors["segment"], size=8.1)
    chip(right_x + 0.262, 0.108, 0.050, 0.032, "graph", fc=colors["graph_soft"], ec=colors["graph"], color=colors["graph"], size=8.1)

    box(0.696, 0.040, 0.215, 0.035, fc=colors["warn_soft"], ec=colors["warn"], lw=1.0, r=0.010, ls=(0, (4, 2)))
    txt(0.804, 0.057, "3D geometry prior is train-time auxiliary only", size=8.1, color=colors["warn"])

    arr(left_x + left_w, left_y + 0.665, mid_x + 0.014, 0.705, color=colors["chain"], lw=1.4)
    arr(left_x + left_w, left_y + 0.290, mid_x + 0.014, 0.555, color=colors["segment"], lw=1.4, rad=0.02)
    arr(left_x + left_w - 0.020, left_y + 0.145, mid_x + 0.014, 0.405, color=colors["graph"], lw=1.4, rad=0.06)
    arr(left_x + left_w - 0.020, left_y + 0.105, mid_x + 0.014, 0.255, color=colors["stats"], lw=1.2, ls=(0, (3, 2)), rad=-0.03)

    for yy in [0.705, 0.555, 0.405, 0.255]:
        arr(mid_x + mid_w, yy, right_x + 0.020, 0.728, color=colors["gate"], lw=1.15, rad=0.02)
    arr(right_x + 0.170, 0.684, right_x + 0.170, 0.658, color=colors["gate"], lw=1.4)
    arr(right_x + 0.170, 0.568, right_x + 0.170, 0.540, color=colors["gate"], lw=1.4)
    arr(right_x + 0.170, 0.418, right_x + 0.170, 0.380, color=colors["gate"], lw=1.4)
    arr(right_x + 0.170, 0.210, right_x + 0.170, 0.181, color=colors["fuse"], lw=1.5)

    txt(0.50, 0.018, "How to draw the segment modality: use a highlighted token window cut from the ordered polymer token stream, not a 3D chain piece.", size=8.8, color=colors["muted"], style="italic")

    save(fig, "fig_msce_pipeline")


def fig_protocol_setup() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "ink": PALETTE["ink"],
        "muted": PALETTE["gray"],
        "line": PALETTE["line"],
        "blue": PALETTE["blue"],
        "blue_soft": PALETTE["blue_soft"],
        "orange": PALETTE["orange"],
        "orange_soft": PALETTE["orange_soft"],
        "green": PALETTE["green"],
        "green_soft": PALETTE["green_soft"],
        "gray_soft": "#F4F6F8",
    }

    def box(x, y, w, h, *, fc, ec, lw=1.3, r=0.018):
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.004,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
        ax.add_patch(patch)
        return patch

    def txt(x, y, s, *, size=10.0, weight=None, color=None, ha="center", va="center", style=None):
        ax.text(
            x,
            y,
            s,
            fontsize=size,
            fontweight=weight,
            color=color or colors["ink"],
            ha=ha,
            va=va,
            style=style,
        )

    def arrow(x1, y1, x2, y2, *, color, lw=1.6):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                mutation_scale=14,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    txt(0.03, 0.95, "Evaluation protocol", ha="left", size=15.0, weight="bold")
    txt(
        0.03,
        0.91,
        "One evaluation run means one fresh split of the development pool plus one fresh training process under the same predefined protocol.",
        ha="left",
        size=9.4,
        color=colors["muted"],
    )

    # Left: registry composition
    box(0.03, 0.12, 0.24, 0.70, fc="#FFFFFF", ec=colors["line"], lw=1.1)
    txt(0.15, 0.79, "Public registry after overlap purge", size=11.6, weight="bold")

    box(0.05, 0.58, 0.20, 0.16, fc=colors["blue_soft"], ec=colors["blue"])
    txt(0.15, 0.68, "Primary pool", size=10.6, weight="bold", color=colors["blue"])
    txt(0.15, 0.63, "7,180 samples", size=9.6)
    txt(0.15, 0.59, "eligible for repeated random splitting", size=8.2, color=colors["muted"])

    box(0.05, 0.37, 0.20, 0.14, fc=colors["orange_soft"], ec=colors["orange"])
    txt(0.15, 0.46, "Supplement", size=10.4, weight="bold", color=colors["orange"])
    txt(0.15, 0.42, "383 samples", size=9.4)
    txt(0.15, 0.385, "always appended to train", size=8.2, color=colors["muted"])

    box(0.05, 0.17, 0.20, 0.14, fc=colors["green_soft"], ec=colors["green"])
    txt(0.15, 0.26, "External holdout", size=10.4, weight="bold", color=colors["green"])
    txt(0.15, 0.22, "302 samples", size=9.4)
    txt(0.15, 0.185, "never used for model selection", size=8.2, color=colors["muted"])

    # Middle: per-run split
    box(0.32, 0.12, 0.34, 0.70, fc=colors["gray_soft"], ec=colors["line"], lw=1.1)
    txt(0.49, 0.79, "Within each run", size=11.6, weight="bold")
    txt(0.49, 0.74, "Repeated stratified random split on $T_g$ bins", size=9.0, color=colors["muted"])

    box(0.36, 0.56, 0.26, 0.12, fc="#FFFFFF", ec=colors["line"], lw=1.0)
    txt(0.49, 0.63, "Step 1: 15% primary-pool test split", size=10.0, weight="bold")
    txt(0.49, 0.585, "StratifiedShuffleSplit on quantile bins", size=8.6, color=colors["muted"])

    box(0.36, 0.38, 0.26, 0.12, fc="#FFFFFF", ec=colors["line"], lw=1.0)
    txt(0.49, 0.45, "Step 2: train / validation split", size=10.0, weight="bold")
    txt(0.49, 0.405, "second stratified split on the remaining 85%", size=8.6, color=colors["muted"])

    box(0.35, 0.16, 0.08, 0.12, fc=colors["blue_soft"], ec=colors["blue"])
    txt(0.39, 0.23, "Train", size=10.0, weight="bold", color=colors["blue"])
    txt(0.39, 0.19, "5,409", size=9.0)

    box(0.45, 0.16, 0.08, 0.12, fc=colors["orange_soft"], ec=colors["orange"])
    txt(0.49, 0.23, "Val", size=10.0, weight="bold", color=colors["orange"])
    txt(0.49, 0.19, "1,077", size=9.0)

    box(0.55, 0.16, 0.08, 0.12, fc=colors["green_soft"], ec=colors["green"])
    txt(0.59, 0.23, "Test", size=10.0, weight="bold", color=colors["green"])
    txt(0.59, 0.19, "1,077", size=9.0)

    txt(0.49, 0.10, "The 383-sample supplement is appended to train after the split.", size=8.4, color=colors["muted"])

    # Right: what a run means
    box(0.71, 0.12, 0.26, 0.70, fc="#FFFFFF", ec=colors["line"], lw=1.1)
    txt(0.84, 0.79, "What one run changes", size=11.6, weight="bold")

    box(0.74, 0.58, 0.20, 0.10, fc=colors["blue_soft"], ec=colors["blue"])
    txt(0.84, 0.63, "data partition", size=10.1, weight="bold", color=colors["blue"])

    box(0.74, 0.43, 0.20, 0.10, fc=colors["orange_soft"], ec=colors["orange"])
    txt(0.84, 0.48, "weight initialization", size=10.1, weight="bold", color=colors["orange"])

    box(0.74, 0.28, 0.20, 0.10, fc=colors["green_soft"], ec=colors["green"])
    txt(0.84, 0.33, "stochastic training behavior", size=10.1, weight="bold", color=colors["green"])

    box(0.74, 0.15, 0.20, 0.08, fc="#FFF8EE", ec="#C8A25F", lw=1.0)
    txt(0.84, 0.19, "not k-fold, not random forest", size=9.0, weight="bold", color="#8A5A12")

    arrow(0.27, 0.59, 0.32, 0.59, color=colors["blue"])
    arrow(0.27, 0.44, 0.32, 0.44, color=colors["orange"])
    arrow(0.27, 0.24, 0.32, 0.24, color=colors["green"])
    arrow(0.66, 0.50, 0.71, 0.50, color=colors["ink"])
    arrow(0.66, 0.32, 0.71, 0.32, color=colors["ink"])

    save(fig, "fig_protocol_setup")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript figures for the MSCE-RCMF-MASD paper line.")
    parser.add_argument(
        "--diag-dir",
        type=str,
        default="",
        help="Optional diagnostic package directory. Defaults to the canonical current 100-run package.",
    )
    args = parser.parse_args()
    diag_dir = resolve_diag_100(args.diag_dir or None)
    # fig_overview() replaced by draw_fig1_overview.py (new design)
    fig_main_results(diag_dir)
    fig_hardcase_evidence(diag_dir)
    fig_cluster_reduction(diag_dir)
    fig_loss_sketch()
    fig_msce_pipeline()
    fig_protocol_setup()
    print(f"Generated figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
