from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "results" / "quantum_comparison" / "all_quantum_experiment_comparison.csv"
WINNERS_PATH = ROOT / "results" / "quantum_matched_sweep" / "quantum_matched_sweep_winners.csv"
PAPER_TEXT_PATH = ROOT / "PredXGBR_results_metrics_summary.txt"
OUTPUT_PATH = ROOT / "PredXGBR_results_metrics_summary_stickers.html"

PAPER_PJME_MAPE = 1.28
PAPER_PJME_R2 = 0.99


def main() -> None:
    comparison = pd.read_csv(COMPARISON_PATH)
    baseline = get_baseline(comparison)
    html = build_html(comparison, baseline)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


def get_baseline(comparison: pd.DataFrame) -> float:
    row = comparison[
        (comparison["comparison_scope"] == "full_test_strong_baseline")
        & (comparison["model"] == "xgboost_full_baseline")
    ]
    if row.empty:
        raise ValueError("Missing xgboost_full_baseline in all_quantum_experiment_comparison.csv")
    return float(row["MAE"].iloc[0])


def build_html(comparison: pd.DataFrame, baseline: float) -> str:
    full_scopes = {
        "full_test_strong_baseline",
        "full_test_hybrid_quantum_postprocess",
        "full_test_postprocess_sweep",
        "full_test_residual",
    }
    main_rows = comparison[comparison["comparison_scope"].isin(full_scopes)].sort_values("MAE")
    support_rows = comparison[~comparison["comparison_scope"].isin(full_scopes)].sort_values(["comparison_scope", "MAE"])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PredXGBR Results Metrics Summary</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 32px;
      color: #1f2933;
      background: #f7f9fb;
    }}
    h1, h2 {{
      margin-bottom: 8px;
    }}
    .meta {{
      color: #52606d;
      margin-bottom: 20px;
    }}
    .baseline {{
      background: #ffffff;
      border-left: 5px solid #4e79a7;
      padding: 12px 16px;
      margin: 18px 0;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      margin: 16px 0 28px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }}
    th, td {{
      border-bottom: 1px solid #dde3ea;
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      background: #e8eef5;
      font-weight: 700;
    }}
    .status {{
      display: inline-block;
      min-width: 74px;
      border-radius: 999px;
      padding: 4px 8px;
      font-weight: 700;
      text-align: center;
      color: #ffffff;
    }}
    .win {{
      background: #168a4a;
    }}
    .lose {{
      background: #c72c2c;
    }}
    .base {{
      background: #4e79a7;
    }}
    .note {{
      color: #52606d;
      font-size: 12px;
      max-width: 420px;
    }}
    .good {{
      color: #168a4a;
      font-weight: 700;
    }}
    .bad {{
      color: #c72c2c;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <h1>PredXGBR Results Metrics Summary</h1>
  <div class="meta">Generated: {datetime.now().isoformat(timespec="seconds")}</div>
  <div class="baseline">
    <strong>Reference baseline:</strong> full XGBoost MAE = {baseline:.6f}<br>
    <strong>Legend:</strong> <span class="status win">✓ WIN</span>
    <span class="status lose">✗ LOSE</span>
    <span class="status base">= BASE</span>
  </div>

  <h2>Main Full-Test Results</h2>
  {table_html(main_rows, baseline)}

  <h2>Comparison Against Paper PJME PredXGBR-1</h2>
  <p class="meta">
    Paper reference from PredXGBR Table 3: PJME PredXGBR Model1 MAPE = {PAPER_PJME_MAPE:.2f}%, R2 = {PAPER_PJME_R2:.2f}.
  </p>
  {paper_comparison_html(main_rows)}

  <h2>Sampled / Matched Supporting Results</h2>
  <p class="meta">These support the discussion but are not the same full-test benchmark.</p>
  {table_html(support_rows, baseline)}

  <h2>Matched Quantum Sweep</h2>
  {matched_sweep_html()}

  <h2>Recommended Paper Text</h2>
  <div class="baseline">
    The hybrid classical-quantum residual correction improves the full XGBoost baseline on the PJME test set,
    reducing MAE from {baseline:.2f} to {best_hybrid_mae(comparison):.2f}.
  </div>
</body>
</html>
"""


def table_html(rows: pd.DataFrame, baseline: float) -> str:
    body = []
    for row in rows.itertuples(index=False):
        mae = float(row.MAE)
        delta = baseline - mae
        status_class, status_text = status(mae, baseline)
        delta_class = "good" if delta > 0 else "bad" if delta < 0 else ""
        body.append(
            "<tr>"
            f"<td><span class=\"status {status_class}\">{status_text}</span></td>"
            f"<td>{escape(str(row.comparison_scope))}</td>"
            f"<td>{escape(str(row.model))}</td>"
            f"<td>{mae:.6f}</td>"
            f"<td>{float(row.RMSE):.6f}</td>"
            f"<td>{float(row.MAPE):.6f}</td>"
            f"<td>{float(row.R2):.6f}</td>"
            f"<td class=\"{delta_class}\">{delta:.6f}</td>"
            f"<td class=\"note\">{escape(str(row.notes))}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Status</th><th>Scope</th><th>Model</th><th>MAE</th><th>RMSE</th>"
        "<th>MAPE</th><th>R2</th><th>MAE Improvement</th><th>Notes</th></tr></thead>"
        "<tbody>"
        + "\n".join(body)
        + "</tbody></table>"
    )


def matched_sweep_html() -> str:
    if not WINNERS_PATH.exists():
        return "<p class=\"meta\">Matched sweep winners file not found.</p>"
    winners = pd.read_csv(WINNERS_PATH)
    wins = int(winners["beats_matched_classical"].sum())
    total = len(winners)
    return (
        f"<div class=\"baseline\"><strong>Hybrid/quantum wins over matched classical:</strong> "
        f"{wins}/{total}</div>"
    )


def paper_comparison_html(rows: pd.DataFrame) -> str:
    selected = rows[
        rows["model"].isin(
            [
                "xgboost_full_baseline",
                "hybrid_classical_quantum_residual_ridge_static",
                "sequential_last_residual_correction",
                "residual_ridge_features",
            ]
        )
    ].copy()
    selected = selected.sort_values("MAPE")
    body = []
    for row in selected.itertuples(index=False):
        mape = float(row.MAPE)
        r2 = float(row.R2)
        mape_class, mape_text = paper_mape_status(mape)
        r2_class, r2_text = paper_r2_status(r2)
        body.append(
            "<tr>"
            f"<td>{escape(str(row.model))}</td>"
            f"<td>{mape:.6f}</td>"
            f"<td><span class=\"status {mape_class}\">{mape_text}</span></td>"
            f"<td>{r2:.6f}</td>"
            f"<td><span class=\"status {r2_class}\">{r2_text}</span></td>"
            f"<td class=\"note\">Compared against paper PJME PredXGBR-1 MAPE {PAPER_PJME_MAPE:.2f}% and R2 {PAPER_PJME_R2:.2f}.</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Model</th><th>MAPE</th><th>MAPE vs Paper</th><th>R2</th><th>R2 vs Paper</th><th>Notes</th></tr></thead>"
        "<tbody>"
        + "\n".join(body)
        + "</tbody></table>"
    )


def paper_mape_status(mape: float) -> tuple[str, str]:
    if mape <= PAPER_PJME_MAPE:
        return "win", "✓ WIN"
    return "lose", "✗ LOSE"


def paper_r2_status(r2: float) -> tuple[str, str]:
    if r2 >= PAPER_PJME_R2:
        return "win", "✓ WIN"
    return "lose", "✗ LOSE"


def status(mae: float, baseline: float) -> tuple[str, str]:
    if abs(mae - baseline) < 1e-6:
        return "base", "= BASE"
    if mae < baseline:
        return "win", "✓ WIN"
    return "lose", "✗ LOSE"


def best_hybrid_mae(comparison: pd.DataFrame) -> float:
    row = comparison[
        (comparison["comparison_scope"] == "full_test_hybrid_quantum_postprocess")
        & (comparison["model"] == "hybrid_classical_quantum_residual_ridge_static")
    ]
    return float(row["MAE"].iloc[0])


def escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
