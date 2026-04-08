"""Generate figures for the GospelVec paper."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VECTOR_DIR = Path(__file__).resolve().parent.parent / "vectors"
PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"


def figure1_layer_accuracy():
    """Generate Figure 1: Layer accuracy curve."""
    with open(VECTOR_DIR / "meta.json") as f:
        meta = json.load(f)

    accuracies = meta["layer_accuracies"]
    best_layer = meta["best_layer"]
    best_acc = meta["best_accuracy"]
    layers = list(range(len(accuracies)))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot accuracy curve
    ax.plot(layers, accuracies, color="#2c3e50", linewidth=2, zorder=3)
    ax.fill_between(layers, accuracies, alpha=0.15, color="#2c3e50")

    # Mark best layer
    ax.scatter([best_layer], [best_acc], color="#c0392b", s=120, zorder=5,
               edgecolors="white", linewidth=2)
    ax.annotate(f"Layer {best_layer}\n{best_acc:.1%}",
                xy=(best_layer, best_acc),
                xytext=(best_layer + 2, best_acc + 0.02),
                fontsize=11, fontweight="bold", color="#c0392b",
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.5))

    # Chance line
    ax.axhline(y=0.25, color="#95a5a6", linestyle="--", linewidth=1, label="Chance (25%)")

    # Steering window
    ax.axvspan(18, 24, alpha=0.08, color="#e74c3c", label="Steering window (layers 18-24)")

    ax.set_xlabel("Decoder Layer", fontsize=13)
    ax.set_ylabel("Gospel Classification Accuracy", fontsize=13)
    ax.set_title("Figure 1. Gospel Identity Readability Across Layers",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, len(accuracies) - 1)
    ax.set_ylim(0.2, 0.7)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = PAPER_DIR / "figure1_layer_accuracy.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()


def figure2_gospel_geometry():
    """Generate Figure 2: Gospel geometry heatmap."""
    with open(VECTOR_DIR / "meta.json") as f:
        meta = json.load(f)

    gospels = meta["gospels"]

    # Hardcoded from extraction results (avoids torch dependency for figure gen)
    sim_matrix = np.array([
        [1.0000, 0.2985, 0.3682, -0.7686],
        [0.2985, 1.0000, 0.3932, -0.6750],
        [0.3682, 0.3932, 1.0000, -0.8070],
        [-0.7686, -0.6750, -0.8070, 1.0000],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))

    # Custom colormap: red for negative, white for zero, blue for positive
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    im = ax.imshow(sim_matrix, cmap="RdBu", norm=norm, aspect="equal")

    # Labels
    labels = [g.capitalize() for g in gospels]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Annotate cells
    for i in range(4):
        for j in range(4):
            val = sim_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    # Synoptic/Johannine annotation
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False,
                                edgecolor="#27ae60", linewidth=2.5,
                                linestyle="--", label="Synoptic cluster"))

    ax.set_title("Figure 2. Gospel Direction Cosine Similarities (Layer 21)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    plt.tight_layout()

    out_path = PAPER_DIR / "figure2_gospel_geometry.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    figure1_layer_accuracy()
    figure2_gospel_geometry()
