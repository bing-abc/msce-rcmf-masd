from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "submission_final"
CHECK_DIR = OUT_DIR / "artwork_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECK_DIR.mkdir(parents=True, exist_ok=True)


def rounded_box(ax, xy, w, h, fc, ec):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    return patch


def draw_polymer_icon(ax, x0, y0, scale=1.0):
    xs = [x0, x0 + 0.05 * scale, x0 + 0.10 * scale, x0 + 0.15 * scale, x0 + 0.20 * scale]
    ys = [y0, y0 + 0.04 * scale, y0 - 0.03 * scale, y0 + 0.05 * scale, y0]
    ax.plot(xs, ys, color="#29444E", lw=3, solid_capstyle="round")
    ax.text(x0 + 0.07 * scale, y0 + 0.07 * scale, "O", fontsize=16, weight="bold", color="#C55A11")
    hexagon = Polygon(
        [
            (x0 + 0.24 * scale, y0 + 0.01 * scale),
            (x0 + 0.27 * scale, y0 + 0.05 * scale),
            (x0 + 0.33 * scale, y0 + 0.05 * scale),
            (x0 + 0.36 * scale, y0 + 0.01 * scale),
            (x0 + 0.33 * scale, y0 - 0.03 * scale),
            (x0 + 0.27 * scale, y0 - 0.03 * scale),
        ],
        closed=True,
        fill=False,
        edgecolor="#29444E",
        linewidth=2.2,
    )
    ax.add_patch(hexagon)


def draw_model_icons(ax, x0, y0):
    items = [
        ("Graph", "#5B8E7D"),
        ("Descriptors", "#E5A24A"),
        ("Chain\ncontext", "#507DBC"),
    ]
    for idx, (label, color) in enumerate(items):
        cx = x0 + idx * 0.060
        circ = Circle((cx, y0), 0.026, facecolor=color, edgecolor="white", lw=1.3)
        ax.add_patch(circ)
        ax.text(cx, y0 - 0.07, label, ha="center", va="top", fontsize=11, color="#1E2A2F")


def draw_result_bars(ax, x0, y0):
    baseline = 29.376
    proposed = 25.152
    max_val = baseline * 1.15
    bar_w = 0.07
    h1 = 0.20 * baseline / max_val
    h2 = 0.20 * proposed / max_val
    ax.add_patch(FancyBboxPatch((x0, y0), bar_w, h1, boxstyle="round,pad=0.005", facecolor="#D89C6A", edgecolor="none"))
    ax.add_patch(FancyBboxPatch((x0 + 0.10, y0), bar_w, h2, boxstyle="round,pad=0.005", facecolor="#5B8E7D", edgecolor="none"))
    ax.text(x0 + bar_w / 2, y0 + h1 + 0.02, f"{baseline:.2f}", ha="center", fontsize=11)
    ax.text(x0 + 0.10 + bar_w / 2, y0 + h2 + 0.02, f"{proposed:.2f}", ha="center", fontsize=11)
    ax.text(x0 + bar_w / 2, y0 - 0.04, "Baseline", ha="center", va="top", fontsize=11)
    ax.text(x0 + 0.10 + bar_w / 2, y0 - 0.04, "Proposed", ha="center", va="top", fontsize=11)


def main() -> None:
    fig = plt.figure(figsize=(16, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("#FBF8F1")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.08),
            0.96,
            0.84,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            facecolor="#FFFDF7",
            edgecolor="#E5DDD0",
            linewidth=1.5,
        )
    )

    boxes = [
        (0.05, 0.18, 0.20, 0.60, "#EAF2F0", "#8BAF9F"),
        (0.29, 0.18, 0.20, 0.60, "#F8F1E4", "#D2AE67"),
        (0.53, 0.18, 0.20, 0.60, "#EEF3F8", "#8AA5C0"),
        (0.77, 0.18, 0.18, 0.60, "#F7EFEA", "#CB9C7B"),
    ]
    for x, y, w, h, fc, ec in boxes:
        rounded_box(ax, (x, y), w, h, fc, ec)

    for x1, x2 in [(0.25, 0.29), (0.49, 0.53), (0.73, 0.77)]:
        ax.add_patch(FancyArrowPatch((x1, 0.48), (x2, 0.48), arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#6E7F86"))

    ax.text(0.50, 0.93, "Bounded difficult-case Tg correction", ha="center", va="center", fontsize=18, weight="bold", color="#18262B")
    ax.text(0.15, 0.72, "Polymer dataset", ha="center", va="center", fontsize=16.5, weight="bold", color="#17323A")
    ax.text(0.15, 0.28, "7563 in-domain\n302 external", ha="center", va="center", fontsize=13, color="#17323A")
    draw_polymer_icon(ax, 0.085, 0.49, scale=0.38)

    ax.text(0.39, 0.72, "Model", ha="center", va="center", fontsize=17, weight="bold", color="#5A3D00")
    ax.text(0.39, 0.31, "Graph + descriptors + chain context\nMSCE + MASD\nDiagnostics", ha="center", va="center", fontsize=12.5, color="#5A3D00")
    draw_model_icons(ax, 0.335, 0.51)

    ax.text(0.63, 0.72, "Hard-case result", ha="center", va="center", fontsize=17, weight="bold", color="#173A58")
    ax.text(0.63, 0.62, "Hard-subgroup MAE", ha="center", va="center", fontsize=13, color="#173A58")
    draw_result_bars(ax, 0.57, 0.35)
    ax.text(
        0.63,
        0.25,
        "29.38 \N{RIGHTWARDS ARROW} 25.15 K\n-14.4%",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        color="#173A58",
    )

    ax.text(0.86, 0.72, "Design use", ha="center", va="center", fontsize=17, weight="bold", color="#623A19")
    ax.text(0.86, 0.50, "Bounded Tg screening", ha="center", va="center", fontsize=13, color="#623A19")
    ax.text(0.86, 0.34, "Aromatic/ether-oxygen gains", ha="center", va="center", fontsize=11.5, color="#623A19")
    ax.text(0.86, 0.22, "External gain bounded", ha="center", va="center", fontsize=11, color="#623A19")

    out_pdf = OUT_DIR / "Graphical_Abstract.pdf"
    out_png = OUT_DIR / "Graphical_Abstract.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    with Image.open(out_png) as img:
        dims = f"{img.width} x {img.height}"
    lines = [
        f"Graphical_Abstract.png dimensions: {dims} px",
        "Minimum requirement: 1328 x 531 px",
        "Title check: image does not contain the literal heading 'Graphical Abstract'",
        "Origin check: scripted Matplotlib artwork; no generative AI was used to create or modify scientific images, raw data, or experimental results.",
        "Readability note: panel text kept to four short blocks for 5 x 13 cm reduction.",
    ]
    (CHECK_DIR / "graphical_abstract_dimensions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
