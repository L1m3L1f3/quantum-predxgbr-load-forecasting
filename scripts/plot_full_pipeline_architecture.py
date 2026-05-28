from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "paper_comparison"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    colors = {
        "data": "#dbeafe",
        "feature": "#e0f2fe",
        "classical": "#e5e7eb",
        "quantum": "#ede9fe",
        "post": "#dcfce7",
        "output": "#fef3c7",
        "compare": "#fee2e2",
    }

    boxes = {
        "data": box(ax, 0.7, 8.6, 3.4, 1.45, "Paper Datasets", "PJM, PJME, PJMW,\nAEP, Dayton", colors["data"]),
        "features": box(ax, 5.0, 8.6, 3.8, 1.45, "Paper-Style Feature Engineering", "Calendar features\nlag, rolling mean/std/min/max\n6, 12, 24 hours", colors["feature"]),
        "split": box(ax, 9.8, 8.6, 3.0, 1.45, "Train/Test Split", "Dataset-specific split dates\nFeature mode: original", colors["feature"]),
        "xgb": box(ax, 5.0, 6.2, 3.8, 1.45, "Local XGBoost Baseline", "1000 estimators\nreg:squarederror\nearly stopping", colors["classical"]),
        "residual": box(ax, 9.8, 6.2, 3.4, 1.45, "Residual Target", "residual = actual - XGBoost\nused for correction models", colors["classical"]),
        "pca": box(ax, 0.7, 4.0, 3.4, 1.45, "Quantum Input Encoder", "StandardScaler -> PCA\nMinMax scaling to [-pi, pi]", colors["quantum"]),
        "vqc": box(ax, 5.0, 4.0, 3.8, 1.45, "VQC / Quantum Circuit", "AngleEmbedding(Y)\nStronglyEntanglingLayers\nPauliZ expectation readout", colors["quantum"]),
        "qfeat": box(ax, 9.8, 4.0, 3.4, 1.45, "Quantum Features", "Z0..Zn expectation values\nGPU: lightning.gpu", colors["quantum"]),
        "hybrid": box(ax, 14.0, 5.0, 3.4, 1.65, "Hybrid Quantum Residual Ridge", "Inputs: classical features\n+ quantum features\n+ XGBoost prediction", colors["post"]),
        "seq": box(ax, 14.0, 7.4, 3.4, 1.45, "Sequential Postprocess", "Previous observed residual\nvalidation-selected correction", colors["post"]),
        "proposed": box(ax, 7.2, 1.6, 4.2, 1.55, "Proposed Method Output", "Best per dataset prediction\nMAPE, R2, MAE, RMSE", colors["output"]),
        "compare": box(ax, 12.6, 1.6, 4.2, 1.55, "Paper Comparison", "Compare against PredXGBR-1\nTable 3 MAPE and R2\nwin/loss stickers and plots", colors["compare"]),
    }

    arrow(ax, boxes["data"], boxes["features"])
    arrow(ax, boxes["features"], boxes["split"])
    arrow(ax, boxes["split"], boxes["xgb"], start_side="bottom", end_side="right")
    arrow(ax, boxes["xgb"], boxes["residual"])
    arrow(ax, boxes["split"], boxes["pca"], start_side="bottom", end_side="top")
    arrow(ax, boxes["pca"], boxes["vqc"])
    arrow(ax, boxes["vqc"], boxes["qfeat"])
    arrow(ax, boxes["qfeat"], boxes["hybrid"])
    arrow(ax, boxes["residual"], boxes["hybrid"])
    arrow(ax, boxes["xgb"], boxes["hybrid"], start_side="right", end_side="left", rad=-0.18)
    arrow(ax, boxes["residual"], boxes["seq"], start_side="right", end_side="left", rad=0.16)
    arrow(ax, boxes["hybrid"], boxes["proposed"], start_side="bottom", end_side="right")
    arrow(ax, boxes["seq"], boxes["proposed"], start_side="bottom", end_side="top", rad=0.12)
    arrow(ax, boxes["xgb"], boxes["proposed"], start_side="bottom", end_side="left", rad=-0.2)
    arrow(ax, boxes["proposed"], boxes["compare"])

    ax.text(
        9,
        10.55,
        "Full Proposed PredXGBR Hybrid Quantum-Enhanced Forecasting Pipeline",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        9,
        0.55,
        "VQC details: classical features are reduced to qubit angles, encoded with AngleEmbedding, passed through StronglyEntanglingLayers, and measured with PauliZ expectation values.",
        ha="center",
        va="center",
        fontsize=11,
        color="#374151",
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_pipeline_architecture.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "full_pipeline_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR / 'full_pipeline_architecture.png'}")


def box(ax, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.4,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.22, body, ha="center", va="center", fontsize=10)
    return (x, y, w, h)


def anchor(rect: tuple[float, float, float, float], side: str) -> tuple[float, float]:
    x, y, w, h = rect
    if side == "left":
        return (x, y + h / 2)
    if side == "right":
        return (x + w, y + h / 2)
    if side == "top":
        return (x + w / 2, y + h)
    if side == "bottom":
        return (x + w / 2, y)
    raise ValueError(side)


def arrow(
    ax,
    source: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    start_side: str = "right",
    end_side: str = "left",
    rad: float = 0.0,
) -> None:
    start = anchor(source, start_side)
    end = anchor(target, end_side)
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.5,
        color="#374151",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


if __name__ == "__main__":
    main()
