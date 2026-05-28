from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "results" / "quantum_comparison" / "all_quantum_experiment_comparison.csv"
MATCHED_SUMMARY_PATH = ROOT / "results" / "quantum_matched_sweep" / "quantum_matched_model_summary.csv"
WINNERS_PATH = ROOT / "results" / "quantum_matched_sweep" / "quantum_matched_sweep_winners.csv"
OUTPUT_PATH = ROOT / "PredXGBR_results_metrics_summary.txt"


def main() -> None:
    comparison = pd.read_csv(COMPARISON_PATH)
    baseline = get_full_baseline_mae(comparison)
    lines = [
        "PredXGBR Results Metrics Summary",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Legend:",
        "[✓] WIN  = MAE is lower than the full XGBoost baseline",
        "[✗] LOSE = MAE is higher than the full XGBoost baseline",
        "[=] BASE = reference XGBoost baseline",
        "",
        f"Reference full XGBoost baseline MAE: {baseline:.6f}",
        "",
        "Main Full-Test Results",
        "----------------------",
    ]

    full_scopes = [
        "full_test_strong_baseline",
        "full_test_hybrid_quantum_postprocess",
        "full_test_postprocess_sweep",
        "full_test_residual",
    ]
    full = comparison[comparison["comparison_scope"].isin(full_scopes)].copy()
    full = full.sort_values(["MAE", "comparison_scope", "model"])
    lines.extend(format_rows(full, baseline))

    lines.extend(
        [
            "",
            "Sampled / Matched Supporting Results",
            "------------------------------------",
            "These are useful supporting experiments, but they are not the same benchmark as the full-test XGBoost result.",
        ]
    )
    supporting = comparison[~comparison["comparison_scope"].isin(full_scopes)].copy()
    supporting = supporting.sort_values(["comparison_scope", "MAE", "model"])
    lines.extend(format_rows(supporting, baseline))

    lines.extend(build_matched_sweep_section())
    lines.extend(build_recommended_paper_text(baseline, comparison))

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


def get_full_baseline_mae(comparison: pd.DataFrame) -> float:
    row = comparison[
        (comparison["comparison_scope"] == "full_test_strong_baseline")
        & (comparison["model"] == "xgboost_full_baseline")
    ]
    if row.empty:
        raise ValueError("Could not find full_test_strong_baseline / xgboost_full_baseline in comparison CSV.")
    return float(row["MAE"].iloc[0])


def format_rows(rows: pd.DataFrame, baseline: float) -> list[str]:
    output = []
    for row in rows.itertuples(index=False):
        mae = float(row.MAE)
        status = status_marker(row.model, mae, baseline)
        improvement = baseline - mae
        output.append(
            f"{status} {row.comparison_scope} | {row.model} | "
            f"MAE={mae:.6f} | RMSE={float(row.RMSE):.6f} | MAPE={float(row.MAPE):.6f} | "
            f"R2={float(row.R2):.6f} | ΔMAE_vs_XGB={improvement:.6f}"
        )
        output.append(f"    Notes: {row.notes}")
    return output


def status_marker(model: str, mae: float, baseline: float) -> str:
    if abs(mae - baseline) < 1e-6:
        return "[=]"
    if mae < baseline:
        return "[✓]"
    return "[✗]"


def build_matched_sweep_section() -> list[str]:
    if not MATCHED_SUMMARY_PATH.exists() or not WINNERS_PATH.exists():
        return []
    summary = pd.read_csv(MATCHED_SUMMARY_PATH)
    winners = pd.read_csv(WINNERS_PATH)
    quantum_wins = int(winners["beats_matched_classical"].sum())
    total = len(winners)
    lines = [
        "",
        "Matched Low-Data Quantum Sweep Details",
        "--------------------------------------",
        f"Hybrid/quantum wins over matched classical: {quantum_wins}/{total}",
        "",
    ]
    for row in summary.sort_values("mean_MAE").itertuples(index=False):
        marker = "[✓]" if int(row.matched_classical_wins) > 0 else "[✗]"
        lines.append(
            f"{marker} {row.model} | runs={int(row.runs)} | mean_MAE={float(row.mean_MAE):.6f} | "
            f"best_MAE={float(row.best_MAE):.6f} | overall_wins={int(row.overall_wins)} | "
            f"matched_classical_wins={int(row.matched_classical_wins)}"
        )
    return lines


def build_recommended_paper_text(baseline: float, comparison: pd.DataFrame) -> list[str]:
    hybrid = comparison[
        (comparison["comparison_scope"] == "full_test_hybrid_quantum_postprocess")
        & (comparison["model"] == "hybrid_classical_quantum_residual_ridge_static")
    ].iloc[0]
    sequential = comparison[
        (comparison["comparison_scope"] == "full_test_postprocess_sweep")
        & (comparison["model"] == "sequential_last_residual_correction")
    ].iloc[0]
    return [
        "",
        "Recommended Paper Wording",
        "-------------------------",
        (
            "The hybrid classical-quantum residual correction improves the full XGBoost baseline "
            f"on the PJME test set, reducing MAE from {baseline:.2f} to {float(hybrid.MAE):.2f} "
            f"(improvement {baseline - float(hybrid.MAE):.2f})."
        ),
        (
            "A rolling one-step residual correction gives the strongest overall result "
            f"(MAE {float(sequential.MAE):.2f}), but it should be reported separately because it uses "
            "the previous observed residual during sequential evaluation."
        ),
    ]


if __name__ == "__main__":
    main()
