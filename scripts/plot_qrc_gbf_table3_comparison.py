from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "qrc_gbf" / "sweep" / "qrc_gbf_sweep_best_by_dataset.csv"
OUTPUT = ROOT / "results" / "qrc_gbf" / "table3_comparison"
DATASET_ORDER = ["PJM", "PJME", "PJMW", "AEP", "Dayton"]
MODEL_ORDER = ["SVM", "RNN", "LSTM", "TCN", "Transformer", "PredXGBR", "QRC-GBF"]


PAPER_ROWS = [
    ("SVM", "PJM", 5.13, 0.96),
    ("SVM", "PJME", 5.80, 0.96),
    ("SVM", "PJMW", 2.80, 0.96),
    ("SVM", "AEP", 6.23, 0.94),
    ("SVM", "Dayton", 7.36, 0.93),
    ("RNN", "PJM", 19.46, 0.92),
    ("RNN", "PJME", 9.49, 0.93),
    ("RNN", "PJMW", 4.28, 0.59),
    ("RNN", "AEP", 7.86, 0.57),
    ("RNN", "Dayton", 12.74, 0.62),
    ("LSTM", "PJM", 19.96, 0.92),
    ("LSTM", "PJME", 9.21, 0.93),
    ("LSTM", "PJMW", 4.70, 0.91),
    ("LSTM", "AEP", 7.00, 0.93),
    ("LSTM", "Dayton", 10.80, 0.92),
    ("TCN", "PJM", 19.46, 0.92),
    ("TCN", "PJME", 7.85, 0.95),
    ("TCN", "PJMW", 3.90, 0.88),
    ("TCN", "AEP", 7.86, 0.57),
    ("TCN", "Dayton", 12.74, 0.62),
    ("Transformer", "PJM", 19.96, 0.92),
    ("Transformer", "PJME", 8.10, 0.94),
    ("Transformer", "PJMW", 4.05, 0.89),
    ("Transformer", "AEP", 7.00, 0.93),
    ("Transformer", "Dayton", 10.80, 0.92),
    ("PredXGBR", "PJM", 1.07, 0.99),
    ("PredXGBR", "PJME", 1.28, 0.99),
    ("PredXGBR", "PJMW", 1.07, 0.98),
    ("PredXGBR", "AEP", 0.98, 0.99),
    ("PredXGBR", "Dayton", 1.12, 0.99),
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paper = pd.DataFrame(PAPER_ROWS, columns=["model", "dataset", "MAPE", "R2"])
    qrc = pd.read_csv(INPUT)
    qrc_rows = qrc.assign(
        model="QRC-GBF",
        dataset=qrc["dataset"].replace({"DAYTON": "Dayton"}),
    )[["model", "dataset", "MAPE", "R2"]]
    combined = pd.concat([paper, qrc_rows], ignore_index=True)
    combined["model"] = pd.Categorical(combined["model"], MODEL_ORDER, ordered=True)
    combined["dataset"] = pd.Categorical(combined["dataset"], DATASET_ORDER, ordered=True)
    combined = combined.sort_values(["model", "dataset"])
    combined.to_csv(OUTPUT / "table3_plus_qrc_gbf_comparison.csv", index=False)

    average = (
        combined.groupby("model", observed=True)
        .agg(average_MAPE=("MAPE", "mean"), average_R2=("R2", "mean"))
        .reset_index()
        .sort_values("average_MAPE")
    )
    average.to_csv(OUTPUT / "table3_plus_qrc_gbf_average_metrics.csv", index=False)

    plot_dataset_mape(combined)
    plot_dataset_r2(combined)
    plot_average_metrics(average)
    plot_heatmap(combined, "MAPE", "table3_qrc_gbf_mape_heatmap.png", "MAPE (%) lower is better", reverse=False)
    plot_heatmap(combined, "R2", "table3_qrc_gbf_r2_heatmap.png", "R2 higher is better", reverse=True)
    write_summary(combined, average)
    print(f"Wrote Table 3 comparison plots to {OUTPUT}")


def plot_dataset_mape(combined: pd.DataFrame) -> None:
    pivot = combined.pivot(index="dataset", columns="model", values="MAPE").loc[DATASET_ORDER, MODEL_ORDER]
    x = np.arange(len(DATASET_ORDER))
    width = 0.11
    colors = ["#8d99ae", "#b8b8d1", "#b0c4b1", "#f4a261", "#90be6d", "#457b9d", "#2a9d8f"]
    plt.figure(figsize=(14, 6))
    for i, model in enumerate(MODEL_ORDER):
        plt.bar(x + (i - 3) * width, pivot[model], width=width, label=model, color=colors[i])
    plt.xticks(x, DATASET_ORDER)
    plt.ylabel("MAPE (%) lower is better")
    plt.title("QRC-GBF vs PredXGBR Table 3 Models: MAPE")
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT / "table3_qrc_gbf_mape_by_dataset.png", dpi=180)
    plt.close()


def plot_dataset_r2(combined: pd.DataFrame) -> None:
    pivot = combined.pivot(index="dataset", columns="model", values="R2").loc[DATASET_ORDER, MODEL_ORDER]
    x = np.arange(len(DATASET_ORDER))
    width = 0.11
    colors = ["#8d99ae", "#b8b8d1", "#b0c4b1", "#f4a261", "#90be6d", "#457b9d", "#2a9d8f"]
    plt.figure(figsize=(14, 6))
    for i, model in enumerate(MODEL_ORDER):
        plt.bar(x + (i - 3) * width, pivot[model], width=width, label=model, color=colors[i])
    plt.xticks(x, DATASET_ORDER)
    plt.ylabel("R2 higher is better")
    plt.ylim(0.5, 1.01)
    plt.title("QRC-GBF vs PredXGBR Table 3 Models: R2")
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT / "table3_qrc_gbf_r2_by_dataset.png", dpi=180)
    plt.close()


def plot_average_metrics(average: pd.DataFrame) -> None:
    ordered = average.sort_values("average_MAPE")
    colors = ["#2a9d8f" if model == "QRC-GBF" else "#8d99ae" for model in ordered["model"].astype(str)]
    plt.figure(figsize=(10, 5))
    plt.bar(ordered["model"].astype(str), ordered["average_MAPE"], color=colors)
    plt.ylabel("Average MAPE (%) lower is better")
    plt.title("Average MAPE Across Five Datasets")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT / "table3_qrc_gbf_average_mape.png", dpi=180)
    plt.close()

    ordered_r2 = average.sort_values("average_R2", ascending=False)
    colors = ["#2a9d8f" if model == "QRC-GBF" else "#8d99ae" for model in ordered_r2["model"].astype(str)]
    plt.figure(figsize=(10, 5))
    plt.bar(ordered_r2["model"].astype(str), ordered_r2["average_R2"], color=colors)
    plt.ylabel("Average R2 higher is better")
    plt.ylim(0.55, 1.01)
    plt.title("Average R2 Across Five Datasets")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT / "table3_qrc_gbf_average_r2.png", dpi=180)
    plt.close()


def plot_heatmap(combined: pd.DataFrame, metric: str, filename: str, title: str, reverse: bool) -> None:
    pivot = combined.pivot(index="model", columns="dataset", values=metric).loc[MODEL_ORDER, DATASET_ORDER]
    data = pivot.to_numpy(dtype=float)
    plt.figure(figsize=(9, 6))
    cmap = "YlGn" if reverse else "YlGn_r"
    plt.imshow(data, aspect="auto", cmap=cmap)
    plt.colorbar(label=metric)
    plt.xticks(np.arange(len(DATASET_ORDER)), DATASET_ORDER)
    plt.yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:.3f}" if metric == "R2" else f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=180)
    plt.close()


def write_summary(combined: pd.DataFrame, average: pd.DataFrame) -> None:
    qrc = combined[combined["model"].astype(str) == "QRC-GBF"].copy()
    pred = combined[combined["model"].astype(str) == "PredXGBR"].copy()
    merged = qrc.merge(pred, on="dataset", suffixes=("_qrc", "_predxgbr"))
    merged["MAPE_improvement_percent"] = (merged["MAPE_predxgbr"] - merged["MAPE_qrc"]) / merged["MAPE_predxgbr"] * 100
    merged[["dataset", "MAPE_predxgbr", "MAPE_qrc", "R2_predxgbr", "R2_qrc", "MAPE_improvement_percent"]].to_csv(
        OUTPUT / "qrc_gbf_vs_predxgbr_table3.csv",
        index=False,
    )
    lines = [
        "QRC-GBF vs PredXGBR Table 3 Comparison",
        "",
        "QRC-GBF is compared against all Model1 values reported in Table 3.",
        "PredXGBR is the strongest paper baseline, so beating PredXGBR implies beating the other Table 3 models on MAPE.",
        "",
        "QRC-GBF vs PredXGBR:",
    ]
    for row in merged.sort_values("dataset").itertuples(index=False):
        lines.append(
            f"{row.dataset}: PredXGBR MAPE={row.MAPE_predxgbr:.4f}, QRC-GBF MAPE={row.MAPE_qrc:.4f}, "
            f"improvement={row.MAPE_improvement_percent:.2f}%, PredXGBR R2={row.R2_predxgbr:.4f}, QRC-GBF R2={row.R2_qrc:.6f}"
        )
    best_mape = average.sort_values("average_MAPE").iloc[0]
    best_r2 = average.sort_values("average_R2", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            f"Best average MAPE model: {best_mape.model} ({best_mape.average_MAPE:.4f})",
            f"Best average R2 model: {best_r2.model} ({best_r2.average_R2:.6f})",
        ]
    )
    (OUTPUT / "table3_qrc_gbf_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
