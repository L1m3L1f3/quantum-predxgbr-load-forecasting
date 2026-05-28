from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


FIGURE_DIR = Path("figures")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    palette = {
        "input": "#dbeafe",
        "prep": "#e0f2fe",
        "xgb": "#e5e7eb",
        "quantum": "#ede9fe",
        "fusion": "#dcfce7",
        "eval": "#fef3c7",
        "paper": "#fee2e2",
        "line": "#1f2937",
    }

    title = "Hybrid Quantum-Enhanced XGBoost for Short-Term Electrical Load Forecasting"
    ax.text(10, 11.55, title, ha="center", va="center", fontsize=22, fontweight="bold")

    data = node(
        ax,
        0.65,
        9.05,
        3.2,
        1.25,
        "Input Datasets",
        "PJM, PJME, PJMW,\nAEP, Dayton hourly load",
        palette["input"],
    )
    clean = node(
        ax,
        4.55,
        9.05,
        3.1,
        1.25,
        "Preprocessing",
        "Datetime parsing\nsorting, train/test split\nmissing-value handling",
        palette["prep"],
    )
    feats = node(
        ax,
        8.35,
        9.05,
        3.45,
        1.25,
        "Feature Engineering",
        "Calendar features\nlag features\nrolling statistics",
        palette["prep"],
    )
    xgb = node(
        ax,
        12.55,
        9.05,
        3.15,
        1.25,
        "XGBoost Baseline",
        "Regression trees\nbaseline forecast\nŷ_XGB(t)",
        palette["xgb"],
    )

    residual = node(
        ax,
        16.45,
        9.05,
        2.9,
        1.25,
        "Residual Signal",
        "r(t) = y(t) - ŷ_XGB(t)\nlearn correction",
        palette["xgb"],
    )

    scaler = node(
        ax,
        1.0,
        6.15,
        3.0,
        1.2,
        "Quantum Pre-Encoder",
        "StandardScaler\nPCA to n qubits\nangle scaling [-π, π]",
        palette["quantum"],
    )
    circuit_box = node(
        ax,
        5.0,
        5.25,
        6.2,
        2.9,
        "Variational Quantum Circuit (VQC)",
        "",
        palette["quantum"],
    )
    draw_vqc(ax, x0=5.35, y0=5.65, width=5.45, height=1.95)

    qreadout = node(
        ax,
        12.25,
        6.15,
        3.25,
        1.2,
        "Quantum Feature Readout",
        "Pauli-Z expectations\nz(t) = [<Z0>,...,<Zn>]\nGPU: lightning.gpu",
        palette["quantum"],
    )
    fusion = node(
        ax,
        16.35,
        6.15,
        3.0,
        1.2,
        "Hybrid Residual Model",
        "Ridge readout over\nclassical + quantum features\nΔŷ_HQ(t)",
        palette["fusion"],
    )

    seq = node(
        ax,
        2.15,
        2.85,
        3.55,
        1.25,
        "Sequential Correction",
        "Previous observed residual\nvalidation-selected α\nrolling one-step mode",
        palette["fusion"],
    )
    pred = node(
        ax,
        7.2,
        2.85,
        3.7,
        1.25,
        "Final Prediction",
        "ŷ(t) = ŷ_XGB(t) + Δŷ_HQ(t)\noptionally + sequential residual",
        palette["fusion"],
    )
    eval_box = node(
        ax,
        12.2,
        2.85,
        3.1,
        1.25,
        "Evaluation",
        "MAPE, R², MAE, RMSE\nper dataset and model",
        palette["eval"],
    )
    compare = node(
        ax,
        16.25,
        2.85,
        3.1,
        1.25,
        "Paper Baseline Comparison",
        "PredXGBR-1 Table 3\nwin/loss by dataset\nplots and tables",
        palette["paper"],
    )

    connect(ax, data, clean)
    connect(ax, clean, feats)
    connect(ax, feats, xgb)
    connect(ax, xgb, residual)

    connect(ax, feats, scaler, start="bottom", end="top", rad=-0.15)
    connect(ax, scaler, circuit_box)
    connect(ax, circuit_box, qreadout)
    connect(ax, qreadout, fusion)
    connect(ax, residual, fusion, start="bottom", end="top", rad=0.08)
    connect(ax, xgb, pred, start="bottom", end="top", rad=-0.25)
    connect(ax, fusion, pred, start="bottom", end="right", rad=0.18)
    connect(ax, residual, seq, start="bottom", end="top", rad=0.22)
    connect(ax, seq, pred)
    connect(ax, pred, eval_box)
    connect(ax, eval_box, compare)

    annotate(ax, 7.95, 4.75, "AngleEmbedding(Y) + trainable/random StronglyEntanglingLayers")
    annotate(ax, 17.85, 5.55, "Static hybrid branch")
    annotate(ax, 3.9, 2.35, "Rolling branch, reported separately")

    footer = (
        "Pipeline used in experiments: original paper-style rolling features, local XGBoost baseline, "
        "VQC quantum features, hybrid residual correction, sequential postprocess, and comparison against PredXGBR-1."
    )
    ax.text(10, 0.65, footer, ha="center", va="center", fontsize=10.5, color="#374151")

    png_path = FIGURE_DIR / "hybrid_quantum_xgboost_architecture.png"
    svg_path = FIGURE_DIR / "hybrid_quantum_xgboost_architecture.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


def node(ax, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.045,rounding_size=0.08",
        linewidth=1.25,
        edgecolor="#111827",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="center", fontsize=11, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h / 2 - 0.2, body, ha="center", va="center", fontsize=9.5)
    return (x, y, w, h)


def draw_vqc(ax, x0: float, y0: float, width: float, height: float) -> None:
    wires = 4
    y_positions = [y0 + height * (i + 1) / (wires + 1) for i in range(wires)]
    x_left = x0
    x_right = x0 + width
    for idx, y in enumerate(y_positions):
        ax.plot([x_left, x_right], [y, y], color="#111827", linewidth=1.2)
        gate(ax, x_left + 0.65, y, "Ry(x)")
        gate(ax, x_left + 1.85, y, "U(θ)")
        gate(ax, x_left + 3.15, y, "U(θ)")
        ax.text(x_right + 0.12, y, f"<Z{idx}>", va="center", fontsize=8.8)

    cnot_xs = [x_left + 2.45, x_left + 3.75, x_left + 4.55]
    for x in cnot_xs:
        for y1, y2 in zip(y_positions[:-1], y_positions[1:]):
            ax.plot([x, x], [y1, y2], color="#4c1d95", linewidth=1.15)
            ax.add_patch(Circle((x, y1), 0.065, color="#4c1d95"))
            ax.add_patch(Circle((x, y2), 0.14, fill=False, edgecolor="#4c1d95", linewidth=1.1))
            ax.plot([x - 0.14, x + 0.14], [y2, y2], color="#4c1d95", linewidth=1.0)
            ax.plot([x, x], [y2 - 0.14, y2 + 0.14], color="#4c1d95", linewidth=1.0)

    ax.text(x0 + width / 2, y0 + height + 0.28, "n-qubit circuit block", ha="center", fontsize=9.5, color="#4c1d95")


def gate(ax, x: float, y: float, label: str) -> None:
    w, h = 0.58, 0.34
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0,
        edgecolor="#4c1d95",
        facecolor="#ffffff",
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color="#4c1d95")


def anchor(rect: tuple[float, float, float, float], side: str) -> tuple[float, float]:
    x, y, w, h = rect
    if side == "left":
        return x, y + h / 2
    if side == "right":
        return x + w, y + h / 2
    if side == "top":
        return x + w / 2, y + h
    if side == "bottom":
        return x + w / 2, y
    raise ValueError(side)


def connect(
    ax,
    source: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    start: str = "right",
    end: str = "left",
    rad: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        anchor(source, start),
        anchor(target, end),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.35,
        color="#374151",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def annotate(ax, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.8,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": "#cbd5e1"},
    )


if __name__ == "__main__":
    main()
