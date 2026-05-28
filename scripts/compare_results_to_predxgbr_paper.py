from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_TEXT_PATH = ROOT / "PredXGBR_results_metrics_summary.txt"
LOCAL_RESULTS_PATH = ROOT / "results" / "quantum_comparison" / "all_quantum_experiment_comparison.csv"
OUTPUT_DIR = ROOT / "results" / "paper_comparison"

PAPER_ROW_PATTERN = re.compile(
    r"^(?P<model>[A-Za-z0-9]+)\s*\|\s*"
    r"(?P<dataset>PJM|PJME|PJMW|AEP|Dayton)\s*\|\s*"
    r"(?P<model1_mape>[0-9.]+)\s*\|\s*"
    r"(?P<model2_mape>[0-9.]+)\s*\|\s*"
    r"(?P<model1_r2>[0-9.]+)\s*\|\s*"
    r"(?P<model2_r2>[0-9.]+)\s*$"
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paper = parse_paper_table(PAPER_TEXT_PATH)
    local = pd.read_csv(LOCAL_RESULTS_PATH)

    paper.to_csv(OUTPUT_DIR / "paper_table3_parsed.csv", index=False)
    local.to_csv(OUTPUT_DIR / "local_results_used_for_comparison.csv", index=False)

    comparison = compare_local_to_paper_pjme(local, paper)
    comparison.to_csv(OUTPUT_DIR / "local_vs_paper_pjme_comparison.csv", index=False)

    paper_only = build_paper_only_dataset_status(paper)
    paper_only.to_csv(OUTPUT_DIR / "paper_datasets_local_availability.csv", index=False)

    html = build_html(comparison, paper_only)
    (OUTPUT_DIR / "local_vs_paper_pjme_comparison_stickers.html").write_text(html, encoding="utf-8")
    (OUTPUT_DIR / "local_vs_paper_pjme_comparison.txt").write_text(build_text_report(comparison, paper_only), encoding="utf-8")
    print(f"Wrote comparison folder: {OUTPUT_DIR}")


def parse_paper_table(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PAPER_ROW_PATTERN.match(line.strip())
        if not match:
            continue
        row = match.groupdict()
        rows.append(
            {
                "paper_model": row["model"],
                "dataset": row["dataset"].upper(),
                "paper_model1_mape": float(row["model1_mape"]),
                "paper_model2_mape": float(row["model2_mape"]),
                "paper_model1_r2": float(row["model1_r2"]),
                "paper_model2_r2": float(row["model2_r2"]),
            }
        )
    if not rows:
        raise ValueError(f"No paper Table 3 rows found in {path}")
    return pd.DataFrame(rows)


def compare_local_to_paper_pjme(local: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    paper_pjme = paper[paper["dataset"] == "PJME"].copy()
    paper_predxgbr = paper_pjme[paper_pjme["paper_model"] == "PredXGBR"].iloc[0]
    rows = []
    for local_row in local[local["dataset"] == "PJME"].itertuples(index=False):
        for paper_row in paper_pjme.itertuples(index=False):
            local_mape = float(local_row.MAPE)
            local_r2 = float(local_row.R2)
            paper_mape = float(paper_row.paper_model1_mape)
            paper_r2 = float(paper_row.paper_model1_r2)
            rows.append(
                {
                    "dataset": "PJME",
                    "local_scope": local_row.comparison_scope,
                    "local_model": local_row.model,
                    "local_MAE": float(local_row.MAE),
                    "local_MAPE": local_mape,
                    "local_R2": local_r2,
                    "paper_model": paper_row.paper_model,
                    "paper_model1_MAPE": paper_mape,
                    "paper_model1_R2": paper_r2,
                    "mape_delta_vs_paper": local_mape - paper_mape,
                    "r2_delta_vs_paper": local_r2 - paper_r2,
                    "beats_paper_mape": local_mape < paper_mape,
                    "beats_or_ties_paper_r2": local_r2 >= paper_r2,
                    "beats_paper_predxgbr_mape": local_mape < float(paper_predxgbr.paper_model1_mape),
                    "beats_or_ties_paper_predxgbr_r2": local_r2 >= float(paper_predxgbr.paper_model1_r2),
                    "notes": local_row.notes,
                }
            )
    return pd.DataFrame(rows).sort_values(["paper_model", "local_MAPE", "local_model"])


def build_paper_only_dataset_status(paper: pd.DataFrame) -> pd.DataFrame:
    local_available = {"PJME"}
    rows = []
    for dataset in sorted(paper["dataset"].unique()):
        rows.append(
            {
                "dataset": dataset,
                "local_data_available": dataset in local_available,
                "status": "available locally" if dataset in local_available else "paper only; local CSV not present",
            }
        )
    return pd.DataFrame(rows)


def build_html(comparison: pd.DataFrame, paper_only: pd.DataFrame) -> str:
    best_local = comparison.sort_values("local_MAPE").drop_duplicates(["local_model", "local_scope"])
    predxgbr_compare = comparison[comparison["paper_model"] == "PredXGBR"].sort_values("local_MAPE")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Local Results vs PredXGBR Paper</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 32px; background: #f7f9fb; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; margin: 16px 0 28px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    th, td {{ border-bottom: 1px solid #dde3ea; padding: 8px 9px; font-size: 13px; text-align: left; vertical-align: top; }}
    th {{ background: #e8eef5; }}
    .status {{ display: inline-block; min-width: 72px; padding: 4px 8px; border-radius: 999px; color: #fff; font-weight: 700; text-align: center; }}
    .win {{ background: #168a4a; }}
    .lose {{ background: #c72c2c; }}
    .base {{ background: #4e79a7; }}
    .note {{ color: #52606d; max-width: 420px; }}
  </style>
</head>
<body>
  <h1>Local Results vs PredXGBR Paper Summary</h1>
  <p>Generated: {datetime.now().isoformat(timespec="seconds")}</p>
  <p>Source paper text: <code>{PAPER_TEXT_PATH.name}</code>. Local dataset available in this repo: PJME only.</p>

  <h2>Dataset Availability</h2>
  {availability_table(paper_only)}

  <h2>Local Models vs Paper PJME PredXGBR-1</h2>
  <p>Paper PJME PredXGBR-1: MAPE 1.28%, R2 0.99. Green tick means local result beats that metric.</p>
  {predxgbr_table(predxgbr_compare)}

  <h2>Local Models vs Every Paper PJME Model1 Row</h2>
  <p>This shows whether each local result beats SVM/RNN/LSTM/TCN/Transformer/PredXGBR from the paper on PJME.</p>
  {all_paper_table(best_local)}
</body>
</html>
"""


def availability_table(df: pd.DataFrame) -> str:
    rows = []
    for row in df.itertuples(index=False):
        cls = "win" if row.local_data_available else "lose"
        label = "✓ FOUND" if row.local_data_available else "✗ MISSING"
        rows.append(f"<tr><td>{row.dataset}</td><td><span class='status {cls}'>{label}</span></td><td>{row.status}</td></tr>")
    return "<table><thead><tr><th>Dataset</th><th>Status</th><th>Notes</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def predxgbr_table(df: pd.DataFrame) -> str:
    rows = []
    for row in df.itertuples(index=False):
        mape_cls, mape_label = marker(row.beats_paper_mape)
        r2_cls, r2_label = marker(row.beats_or_ties_paper_r2)
        rows.append(
            "<tr>"
            f"<td>{escape(row.local_scope)}</td><td>{escape(row.local_model)}</td>"
            f"<td>{row.local_MAPE:.6f}</td><td><span class='status {mape_cls}'>{mape_label}</span></td>"
            f"<td>{row.local_R2:.6f}</td><td><span class='status {r2_cls}'>{r2_label}</span></td>"
            f"<td class='note'>{escape(str(row.notes))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Scope</th><th>Local Model</th><th>Local MAPE</th><th>MAPE vs Paper PredXGBR</th><th>Local R2</th><th>R2 vs Paper PredXGBR</th><th>Notes</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def all_paper_table(df: pd.DataFrame) -> str:
    rows = []
    for row in df.itertuples(index=False):
        mape_cls, mape_label = marker(row.beats_paper_mape)
        r2_cls, r2_label = marker(row.beats_or_ties_paper_r2)
        rows.append(
            "<tr>"
            f"<td>{escape(row.paper_model)}</td><td>{row.paper_model1_MAPE:.2f}</td><td>{row.paper_model1_R2:.2f}</td>"
            f"<td>{escape(row.local_model)}</td><td>{row.local_MAPE:.6f}</td><td><span class='status {mape_cls}'>{mape_label}</span></td>"
            f"<td>{row.local_R2:.6f}</td><td><span class='status {r2_cls}'>{r2_label}</span></td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Paper Model</th><th>Paper MAPE</th><th>Paper R2</th><th>Local Model</th><th>Local MAPE</th><th>MAPE Status</th><th>Local R2</th><th>R2 Status</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def build_text_report(comparison: pd.DataFrame, paper_only: pd.DataFrame) -> str:
    lines = [
        "Local Results vs PredXGBR Paper Comparison",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Paper source text: {PAPER_TEXT_PATH.name}",
        "",
        "Dataset availability:",
    ]
    for row in paper_only.itertuples(index=False):
        marker_text = "[✓]" if row.local_data_available else "[✗]"
        lines.append(f"{marker_text} {row.dataset}: {row.status}")

    lines.extend(["", "Local models vs paper PJME PredXGBR-1 (MAPE 1.28%, R2 0.99):"])
    predxgbr = comparison[comparison["paper_model"] == "PredXGBR"].sort_values("local_MAPE")
    for row in predxgbr.itertuples(index=False):
        mape_marker = "[✓]" if row.beats_paper_mape else "[✗]"
        r2_marker = "[✓]" if row.beats_or_ties_paper_r2 else "[✗]"
        lines.append(
            f"{mape_marker} MAPE / {r2_marker} R2 | {row.local_scope} | {row.local_model} | "
            f"MAPE={row.local_MAPE:.6f} | R2={row.local_R2:.6f} | notes={row.notes}"
        )
    return "\n".join(lines) + "\n"


def marker(value: bool) -> tuple[str, str]:
    return ("win", "✓ WIN") if value else ("lose", "✗ LOSE")


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    main()
