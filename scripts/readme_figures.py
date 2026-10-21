"""Figures for the README, drawn from the web page's exported data (site/data).

    python scripts/readme_figures.py      # needs matplotlib and pillow

Writes docs/gallery.png (before/after examples) and docs/antakya.png (model vs expert map).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data"
OUT = ROOT / "docs"
CLASSES = ["intact", "damaged", "destroyed"]
COLORS = {"intact": "#8a9a8b", "damaged": "#d0902a", "destroyed": "#a8352a"}
PAPER, INK = "#f6f3ec", "#1d1b18"


def predicted(probs, t, bias):
    """Same decision rule as site/app.js: temperature-adjusted log-probs plus class offsets."""
    s = t * np.log(np.clip(probs, 1e-6, None)) + np.asarray(bias)
    return s.argmax(1)


def gallery(items):
    # one of each case worth showing, in reading order
    cases = [
        ("destroyed", "destroyed"),
        ("destroyed", "destroyed"),
        ("intact", "intact"),
        ("damaged", "damaged"),
        ("destroyed", "intact"),
        ("intact", "destroyed"),
    ]
    picked = []
    for true, pred in cases:
        g = next(
            (g for g in items if g["true"] == true and g["pred"] == pred and g not in picked), None
        )
        if g:
            picked.append(g)
    fig, axes = plt.subplots(2, len(picked), figsize=(2.3 * len(picked), 5.2), facecolor=PAPER)
    for j, g in enumerate(picked):
        fp = np.asarray(Image.open(DATA / "gallery" / f"{g['id']}_fp.png").convert("L")) > 127
        for i, kind in enumerate(("pre", "post")):
            ax = axes[i, j]
            ax.imshow(Image.open(DATA / "gallery" / f"{g['id']}_{kind}.jpg"))
            ax.contour(fp, levels=[0.5], colors="white", linewidths=0.8)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel("before" if kind == "pre" else "after", color=INK, fontsize=11)
        ok = g["true"] == g["pred"]
        axes[1, j].set_xlabel(
            f"label: {g['true']}\nmodel: {g['pred']} ({max(g['probs']):.0%})",
            color=INK if ok else COLORS["destroyed"],
            fontsize=9.5,
        )
    fig.tight_layout()
    fig.savefig(OUT / "gallery.png", dpi=110, facecolor=PAPER)
    plt.close(fig)


def antakya(buildings, t, bias):
    b = np.asarray(buildings, dtype=float)
    pred = predicted(b[:, 3:6], t, bias)
    true = b[:, 2].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), facecolor=PAPER)
    for ax, cls, title in ((axes[0], pred, "Model prediction"), (axes[1], true, "Expert label")):
        ax.set_facecolor(PAPER)
        for c in (0, 1, 2):  # rare classes drawn last so they stay visible
            m = cls == c
            ax.scatter(
                b[m, 0],
                b[m, 1],
                s=2.2 if c == 0 else 3.5,
                c=COLORS[CLASSES[c]],
                alpha=0.45 if c == 0 else 0.9,
                linewidths=0,
                label=CLASSES[c],
            )
        ax.set_title(title, color=INK, fontsize=13, loc="left")
        ax.set_aspect(1 / np.cos(np.deg2rad(b[:, 1].mean())))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[1].legend(loc="lower right", frameon=False, markerscale=4)
    fig.suptitle(
        f"Antakya (held out): {len(b):,} buildings", x=0.02, ha="left", color=INK, fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "antakya.png", dpi=110, facecolor=PAPER)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    res = json.loads((DATA / "results.json").read_text())
    best = res[res["map_model"]]["test_bias"]
    gallery(json.loads((DATA / "gallery.json").read_text()))
    antakya(
        json.loads((DATA / "buildings.json").read_text()),
        best["temperature"],
        best["bias"] or [0, 0, 0],
    )
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
