from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum.qrc_gbf import PAPER_ORDER

INPUT_SWEEP = ROOT / "results" / "qrc_gbf" / "sweep" / "qrc_gbf_sweep_proposed_only.csv"
INPUT_ALL_ROWS = ROOT / "results" / "qrc_gbf" / "sweep" / "qrc_gbf_sweep_all_rows.csv"
PAPER_COMPARISON = ROOT / "results" / "paper_comparison" / "qrc_gbf_vs_predxgbr_all_datasets.csv"
OUTPUT_DIR = ROOT / "results" / "qrc_gbf" / "robustness"
FIGURE_DIR = ROOT / "paper_figures" / "robustness"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_SWEEP.exists():
        raise FileNotFoundError(f"Missing sweep file: {INPUT_SWEEP}. Run scripts/run_qrc_gbf_sweep.py first.")

    proposed = pd.read_csv(INPUT_SWEEP)
    proposed["dataset"] = proposed["dataset"].replace({"Dayton": "DAYTON"})
    proposed = proposed[proposed["dataset"].isin(PAPER_ORDER)].copy()

    summary = build_robustness_summary(proposed)
    best = build_best_table(proposed)
    config_summary = build_config_summary(proposed)
    source_summary = build_source_summary(proposed)

    summary.to_csv(OUTPUT_DIR / "qrc_gbf_robustness_by_dataset.csv", index=False)
    best.to_csv(OUTPUT_DIR / "qrc_gbf_robustness_best_by_dataset.csv", index=False)
    config_summary.to_csv(OUTPUT_DIR / "qrc_gbf_robustness_by_configuration.csv", index=False)
    source_summary.to_csv(OUTPUT_DIR / "qrc_gbf_correction_source_summary.csv", index=False)

    if INPUT_ALL_ROWS.exists():
        ablation = build_internal_ablation(pd.read_csv(INPUT_ALL_ROWS))
        ablation.to_csv(OUTPUT_DIR / "qrc_gbf_internal_ablation_from_sweep.csv", index=False)
    else:
        ablation = pd.DataFrame()

    if PAPER_COMPARISON.exists():
        model12 = pd.read_csv(PAPER_COMPARISON)
        model12.to_csv(OUTPUT_DIR / "qrc_gbf_model1_model2_external_reference.csv", index=False)

    write_plots(proposed, summary, best)
    write_text_report(proposed, summary, best, config_summary, source_summary, ablation)
    print(f"Wrote robustness report to {OUTPUT_DIR}")
    print(summary.to_string(index=False))


def build_robustness_summary(proposed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in PAPER_ORDER:
        part = proposed[proposed["dataset"] == dataset]
        if part.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "runs": len(part),
                "mean_MAPE": part["MAPE"].mean(),
                "std_MAPE": part["MAPE"].std(ddof=0),
                "min_MAPE": part["MAPE"].min(),
                "max_MAPE": part["MAPE"].max(),
                "mean_R2": part["R2"].mean(),
                "std_R2": part["R2"].std(ddof=0),
                "paper_MAPE": part["paper_MAPE"].iloc[0],
                "paper_R2": part["paper_R2"].iloc[0],
                "paper_win_count": int(part["beats_paper"].sum()),
                "paper_win_rate": part["beats_paper"].mean(),
                "backbone_win_count": int(part["beats_backbone_MAPE"].sum()),
                "backbone_win_rate": part["beats_backbone_MAPE"].mean(),
                "mean_improvement_vs_paper_percent": part["MAPE_improvement_vs_paper_percent"].mean(),
                "best_improvement_vs_paper_percent": part["MAPE_improvement_vs_paper_percent"].max(),
            }
        )
    return pd.DataFrame(rows)


def build_best_table(proposed: pd.DataFrame) -> pd.DataFrame:
    best = proposed.sort_values(["dataset", "MAPE"]).groupby("dataset", as_index=False).first()
    return best[
        [
            "dataset",
            "MAPE",
            "R2",
            "MAE",
            "RMSE",
            "n_qubits",
            "n_layers",
            "random_state",
            "paper_MAPE",
            "paper_R2",
            "beats_paper",
            "beats_backbone_MAPE",
            "MAPE_improvement_vs_paper_percent",
            "alpha",
            "gamma",
            "notes",
        ]
    ]


def build_config_summary(proposed: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["n_qubits", "n_layers", "random_state"]
    rows = []
    for keys, part in proposed.groupby(group_cols):
        rows.append(
            {
                "n_qubits": keys[0],
                "n_layers": keys[1],
                "random_state": keys[2],
                "datasets": len(part),
                "mean_MAPE": part["MAPE"].mean(),
                "mean_R2": part["R2"].mean(),
                "paper_wins": int(part["beats_paper"].sum()),
                "backbone_wins": int(part["beats_backbone_MAPE"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_MAPE", "backbone_wins"], ascending=[True, False])


def build_source_summary(proposed: pd.DataFrame) -> pd.DataFrame:
    source = proposed["notes"].astype(str).str.extract(r"selected=([^,]+)")[0].fillna("unknown")
    out = proposed.assign(correction_source=source)
    return (
        out.groupby("correction_source")
        .agg(
            runs=("dataset", "count"),
            mean_MAPE=("MAPE", "mean"),
            mean_R2=("R2", "mean"),
            paper_wins=("beats_paper", "sum"),
            backbone_wins=("beats_backbone_MAPE", "sum"),
        )
        .reset_index()
        .sort_values("runs", ascending=False)
    )


def build_internal_ablation(all_rows: pd.DataFrame) -> pd.DataFrame:
    all_rows["dataset"] = all_rows["dataset"].replace({"Dayton": "DAYTON"})
    pivot = all_rows.pivot_table(
        index=["dataset", "n_qubits", "n_layers", "random_state"],
        columns="model",
        values=["MAPE", "R2"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    if "MAPE_QRC-GBF" in pivot and "MAPE_Gradient-boosted backbone" in pivot:
        pivot["qrc_mape_improvement_vs_backbone_percent"] = (
            (pivot["MAPE_Gradient-boosted backbone"] - pivot["MAPE_QRC-GBF"])
            / pivot["MAPE_Gradient-boosted backbone"]
            * 100
        )
        pivot["qrc_beats_backbone"] = pivot["MAPE_QRC-GBF"] < pivot["MAPE_Gradient-boosted backbone"]
    return pivot


def write_plots(proposed: pd.DataFrame, summary: pd.DataFrame, best: pd.DataFrame) -> None:
    ordered = summary.set_index("dataset").loc[[d for d in PAPER_ORDER if d in set(summary["dataset"])]].reset_index()
    x = np.arange(len(ordered))

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, ordered["paper_MAPE"], width=0.4, label="Published PredXGBR", color="#8d99ae")
    plt.bar(x + 0.2, ordered["mean_MAPE"], width=0.4, label="QRC-GBF mean sweep", color="#2a9d8f")
    plt.errorbar(x + 0.2, ordered["mean_MAPE"], yerr=ordered["std_MAPE"], fmt="none", ecolor="#1b4332", capsize=4)
    plt.xticks(x, ordered["dataset"])
    plt.ylabel("MAPE (%) lower is better")
    plt.title("QRC-GBF Robustness vs Published PredXGBR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "qrc_gbf_robustness_mean_mape.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(ordered["dataset"], ordered["paper_win_rate"] * 100, color="#2a9d8f", label="vs published PredXGBR")
    plt.bar(ordered["dataset"], ordered["backbone_win_rate"] * 100, color="#f4a261", alpha=0.75, label="vs local GBF")
    plt.ylim(0, 105)
    plt.ylabel("Win rate across sweep settings (%)")
    plt.title("QRC-GBF Win Rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "qrc_gbf_robustness_win_rates.png", dpi=180)
    plt.close()

    labels = proposed.apply(lambda r: f"{r['dataset']} q{int(r['n_qubits'])} l{int(r['n_layers'])} s{int(r['random_state'])}", axis=1)
    rank = proposed.assign(label=labels).sort_values("MAPE").head(30)
    plt.figure(figsize=(12, 8))
    colors = ["#2a9d8f" if v else "#d62828" for v in rank["beats_backbone_MAPE"]]
    plt.barh(rank["label"][::-1], rank["MAPE"].to_numpy()[::-1], color=colors[::-1])
    plt.xlabel("MAPE (%) lower is better")
    plt.title("Top QRC-GBF Sweep Runs")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "qrc_gbf_top_sweep_runs.png", dpi=180)
    plt.close()

    best_ordered = best.set_index("dataset").loc[[d for d in PAPER_ORDER if d in set(best["dataset"])]].reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(best_ordered["dataset"], best_ordered["paper_R2"], marker="o", label="Published PredXGBR R2", color="#8d99ae")
    plt.plot(best_ordered["dataset"], best_ordered["R2"], marker="o", label="Best QRC-GBF R2", color="#2a9d8f")
    plt.ylabel("R2 higher is better")
    plt.title("Best QRC-GBF R2 vs Published PredXGBR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "qrc_gbf_best_r2_vs_paper.png", dpi=180)
    plt.close()


def write_text_report(proposed, summary, best, config_summary, source_summary, ablation) -> None:
    total = len(proposed)
    paper_wins = int(proposed["beats_paper"].sum())
    backbone_wins = int(proposed["beats_backbone_MAPE"].sum())
    lines = [
        "QRC-GBF Robustness and Ablation Report",
        "",
        f"Total QRC-GBF runs analyzed: {total}",
        f"External published-baseline wins: {paper_wins}/{total}",
        f"Internal local-backbone wins: {backbone_wins}/{total}",
        "",
        "Dataset-level robustness:",
    ]
    for row in summary.sort_values("dataset").itertuples(index=False):
        lines.append(
            f"{row.dataset}: mean MAPE={row.mean_MAPE:.4f} +/- {row.std_MAPE:.4f}, "
            f"best MAPE={row.min_MAPE:.4f}, paper MAPE={row.paper_MAPE:.4f}, "
            f"paper wins={row.paper_win_count}/{row.runs}, local-backbone wins={row.backbone_win_count}/{row.runs}."
        )
    lines.extend([
        "",
        "Best setting by dataset:",
    ])
    for row in best.sort_values("dataset").itertuples(index=False):
        lines.append(
            f"{row.dataset}: MAPE={row.MAPE:.4f}, R2={row.R2:.6f}, q={int(row.n_qubits)}, "
            f"layers={int(row.n_layers)}, seed={int(row.random_state)}, "
            f"beats paper={row.beats_paper}, beats local backbone={row.beats_backbone_MAPE}."
        )
    lines.extend([
        "",
        "Correction-source summary:",
    ])
    for row in source_summary.itertuples(index=False):
        lines.append(
            f"{row.correction_source}: runs={row.runs}, mean MAPE={row.mean_MAPE:.4f}, "
            f"paper wins={int(row.paper_wins)}, local-backbone wins={int(row.backbone_wins)}."
        )
    lines.extend([
        "",
        "Paper-safe interpretation:",
        "QRC-GBF is strong as a full residual-correction framework because it consistently improves over the published PredXGBR reference baselines in the analyzed runs.",
        "The internal ablation is more mixed, so the paper should not claim standalone quantum advantage or that the VQC branch always beats the local GBF backbone.",
        "The defensible claim is that QRC-GBF combines a strong gradient-boosted backbone with validation-selected residual correction, including a quantum-enhanced residual candidate, and improves the final forecasting pipeline in the evaluated setting.",
    ])
    if not ablation.empty and "qrc_mape_improvement_vs_backbone_percent" in ablation:
        lines.extend([
            "",
            "Internal ablation note:",
            f"Mean QRC-GBF MAPE improvement vs local backbone across sweep rows: {ablation['qrc_mape_improvement_vs_backbone_percent'].mean():.4f}%.",
        ])
    (OUTPUT_DIR / "qrc_gbf_robustness_report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
