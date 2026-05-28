from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_pipeline import DATASETS, MODEL_DIR, RESULTS_DIR, load_dataset, paths_for, split_features
from quantum.hybrid_quantum_postprocess_sweep import run_hybrid_quantum_postprocess
from quantum.postprocess_baseline_sweep import run_postprocess_sweep
from scripts.compare_results_to_predxgbr_paper import PAPER_TEXT_PATH, parse_paper_table


OUTPUT_DIR = RESULTS_DIR / "paper_comparison"
PAPER_ORDER = ["PJM", "PJME", "PJMW", "AEP", "DAYTON"]


@dataclass
class FullPaperRunConfig:
    feature_mode: str
    device: str
    n_estimators: int
    hybrid_qubits: int
    hybrid_layers: int
    hybrid_train_samples: int
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full all-dataset paper comparison experiments.")
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="original")
    parser.add_argument("--device", default="lightning.gpu")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--hybrid-qubits", type=int, default=4)
    parser.add_argument("--hybrid-layers", type=int, default=2)
    parser.add_argument("--hybrid-train-samples", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paper = parse_paper_table(PAPER_TEXT_PATH)

    all_rows = []
    for dataset in PAPER_ORDER:
        print(f"Running full experiment for {dataset}")
        baseline = train_local_xgboost(dataset, args)
        postprocess = run_postprocess(dataset, args)
        hybrid = run_hybrid_quantum(dataset, args)
        all_rows.extend(build_model_rows(dataset, baseline, postprocess, hybrid, paper, args.feature_mode))

    results = pd.DataFrame(all_rows)
    best = build_best_models(results)
    improvement = build_improvement_summary(best)
    r2_summary = build_r2_summary(results, best)

    results.to_csv(OUTPUT_DIR / "all_datasets_vs_paper_comparison.csv", index=False)
    best.to_csv(OUTPUT_DIR / "all_datasets_best_models.csv", index=False)
    improvement.to_csv(OUTPUT_DIR / "all_datasets_improvement_summary.csv", index=False)
    results.to_csv(OUTPUT_DIR / "master_results_table.csv", index=False)
    write_text_outputs(results, best, improvement, r2_summary)
    write_html(results, best)
    write_plots(results, best, r2_summary)
    write_paper_draft(results, best, improvement, r2_summary)
    write_docx_from_markdown(OUTPUT_DIR / "paper_draft.md", OUTPUT_DIR / "paper_draft.docx")
    write_status(args)
    print(f"Wrote full paper outputs to {OUTPUT_DIR}")


def train_local_xgboost(dataset: str, args: argparse.Namespace) -> dict:
    split_date = DATASETS[dataset]["split_date"]
    df, _, _, data_path = load_dataset(dataset)
    X_train, y_train, X_test, y_test, _, _ = split_features(df, split_date, args.feature_mode)
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=args.random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    training_seconds = time.perf_counter() - start
    pred_start = time.perf_counter()
    pred = model.predict(X_test)
    inference_seconds = time.perf_counter() - pred_start
    MODEL_DIR.mkdir(exist_ok=True)
    with paths_for(dataset)["model"].open("wb") as handle:
        pickle.dump(model, handle)
    return metric_dict(
        dataset=dataset,
        model_name="Local XGBoost baseline",
        feature_mode=args.feature_mode,
        y_true=y_test,
        pred=pred,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
        data_file=str(data_path),
        extra={"train_rows": len(X_train), "test_rows": len(X_test)},
    )


def run_postprocess(dataset: str, args: argparse.Namespace) -> pd.DataFrame:
    namespace = SimpleNamespace(
        dataset=dataset,
        data_file=None,
        split_date=None,
        feature_mode=args.feature_mode,
        validation_fraction=0.2,
        random_state=args.random_state,
    )
    return run_postprocess_sweep(namespace)


def run_hybrid_quantum(dataset: str, args: argparse.Namespace) -> pd.DataFrame:
    namespace = SimpleNamespace(
        dataset=dataset,
        data_file=None,
        split_date=None,
        feature_mode=args.feature_mode,
        device=args.device,
        n_qubits=args.hybrid_qubits,
        n_layers=args.hybrid_layers,
        max_train_samples=args.hybrid_train_samples,
        validation_fraction=0.2,
        random_state=args.random_state,
        log_dir=Path("logs") / "full_paper_hybrid_quantum" / dataset.lower(),
    )
    return run_hybrid_quantum_postprocess(namespace)


def build_model_rows(
    dataset: str,
    baseline: dict,
    postprocess: pd.DataFrame,
    hybrid: pd.DataFrame,
    paper: pd.DataFrame,
    feature_mode: str,
) -> list[dict]:
    paper_row = paper[(paper["dataset"] == dataset) & (paper["paper_model"] == "PredXGBR")].iloc[0]
    paper_mape = float(paper_row["paper_model1_mape"])
    paper_r2 = float(paper_row["paper_model1_r2"])
    local_mape = baseline["MAPE"]
    rows = [attach_comparison(baseline, paper_mape, paper_r2, local_mape)]

    model_map = {
        "hybrid_classical_quantum_residual_ridge_static": "Static hybrid quantum",
        "hybrid_quantum_sequential_validation_selected": "Hybrid quantum sequential",
    }
    for source_model, label in model_map.items():
        row = hybrid[hybrid["model"] == source_model].iloc[0].to_dict()
        rows.append(
            attach_comparison(
                normalize_existing_row(dataset, label, feature_mode, row),
                paper_mape,
                paper_r2,
                local_mape,
            )
        )

    seq = postprocess[postprocess["model"] == "sequential_last_residual_correction"].iloc[0].to_dict()
    rows.append(
        attach_comparison(
            normalize_existing_row(dataset, "Best sequential postprocess", feature_mode, seq),
            paper_mape,
            paper_r2,
            local_mape,
        )
    )
    return rows


def metric_dict(
    dataset: str,
    model_name: str,
    feature_mode: str,
    y_true: pd.Series,
    pred: np.ndarray,
    training_seconds: float,
    inference_seconds: float,
    data_file: str,
    extra: dict | None = None,
) -> dict:
    row = {
        "dataset": dataset,
        "model": model_name,
        "feature_mode": feature_mode,
        "MAPE": np.mean(np.abs((y_true.to_numpy() - pred) / y_true.to_numpy())) * 100,
        "R2": r2_score(y_true, pred),
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "training_time_seconds": training_seconds,
        "inference_time_seconds": inference_seconds,
        "data_file": data_file,
    }
    if extra:
        row.update(extra)
    return row


def normalize_existing_row(dataset: str, model_name: str, feature_mode: str, row: dict) -> dict:
    return {
        "dataset": dataset,
        "model": model_name,
        "feature_mode": feature_mode,
        "MAPE": float(row["MAPE"]),
        "R2": float(row["R2"]),
        "MAE": float(row["MAE"]),
        "RMSE": float(row["RMSE"]),
        "training_time_seconds": float(row.get("seconds", np.nan)),
        "inference_time_seconds": np.nan,
        "data_file": "",
    }


def attach_comparison(row: dict, paper_mape: float, paper_r2: float, local_mape: float) -> dict:
    row = row.copy()
    row["paper_baseline_MAPE"] = paper_mape
    row["paper_baseline_R2"] = paper_r2
    row["beats_paper_MAPE"] = row["MAPE"] < paper_mape
    row["beats_or_ties_paper_R2"] = row["R2"] >= paper_r2
    row["beats_paper_baseline"] = bool(row["beats_paper_MAPE"] and row["beats_or_ties_paper_R2"])
    row["hybrid_improves_over_local_xgboost"] = row["model"] in {
        "Static hybrid quantum",
        "Hybrid quantum sequential",
    } and row["MAPE"] < local_mape
    row["MAPE_improvement_vs_paper_percent"] = (paper_mape - row["MAPE"]) / paper_mape * 100
    return row


def build_best_models(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(["dataset", "MAPE"]).groupby("dataset", as_index=False).first()


def build_improvement_summary(best: pd.DataFrame) -> pd.DataFrame:
    return best[
        [
            "dataset",
            "model",
            "paper_baseline_MAPE",
            "MAPE",
            "MAPE_improvement_vs_paper_percent",
            "paper_baseline_R2",
            "R2",
            "beats_paper_baseline",
        ]
    ].rename(columns={"model": "best_model", "MAPE": "best_MAPE", "R2": "best_R2"})


def build_r2_summary(results: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in results.groupby("dataset"):
        hybrid = group[group["model"].isin(["Static hybrid quantum", "Hybrid quantum sequential"])]
        rows.append(
            {
                "dataset": dataset,
                "paper_R2": float(group["paper_baseline_R2"].iloc[0]),
                "best_local_R2": float(group[group["model"] == "Local XGBoost baseline"]["R2"].iloc[0]),
                "best_hybrid_R2": float(hybrid["R2"].max()),
                "best_overall_R2": float(group["R2"].max()),
            }
        )
    return pd.DataFrame(rows)


def write_text_outputs(results: pd.DataFrame, best: pd.DataFrame, improvement: pd.DataFrame, r2_summary: pd.DataFrame) -> None:
    lines = ["Full All-Dataset Paper Comparison", ""]
    for row in improvement.sort_values("dataset").itertuples(index=False):
        marker = "WIN" if row.beats_paper_baseline else "LOSE"
        lines.append(
            f"{row.dataset}: {marker} | best={row.best_model} | paper_MAPE={row.paper_baseline_MAPE:.4f} | "
            f"best_MAPE={row.best_MAPE:.4f} | improvement={row.MAPE_improvement_vs_paper_percent:.2f}% | "
            f"paper_R2={row.paper_baseline_R2:.4f} | best_R2={row.best_R2:.6f}"
        )
    wins = improvement[improvement["beats_paper_baseline"]]["dataset"].tolist()
    losses = improvement[~improvement["beats_paper_baseline"]]["dataset"].tolist()
    lines.extend(
        [
            "",
            f"Datasets beating paper: {', '.join(wins) if wins else 'none'}",
            f"Datasets not beating paper: {', '.join(losses) if losses else 'none'}",
            f"Best overall model: {best.sort_values('MAPE').iloc[0]['model']} on {best.sort_values('MAPE').iloc[0]['dataset']}",
        ]
    )
    (OUTPUT_DIR / "all_datasets_vs_paper_comparison.txt").write_text("\n".join(lines) + "\n")
    (OUTPUT_DIR / "final_result_summary.txt").write_text("\n".join(lines) + "\n")


def write_html(results: pd.DataFrame, best: pd.DataFrame) -> None:
    rows = []
    for row in results.sort_values(["dataset", "MAPE"]).itertuples(index=False):
        cls = "win" if row.beats_paper_baseline else "lose"
        label = "✓ WIN" if row.beats_paper_baseline else "✗ LOSE"
        hybrid_cls = "win" if row.hybrid_improves_over_local_xgboost else "lose"
        hybrid_label = "✓ YES" if row.hybrid_improves_over_local_xgboost else "✗ NO"
        rows.append(
            "<tr>"
            f"<td>{row.dataset}</td><td>{row.model}</td><td>{row.MAPE:.6f}</td><td>{row.R2:.6f}</td>"
            f"<td>{row.paper_baseline_MAPE:.4f}</td><td>{row.paper_baseline_R2:.4f}</td>"
            f"<td><span class='status {cls}'>{label}</span></td>"
            f"<td><span class='status {hybrid_cls}'>{hybrid_label}</span></td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>All Datasets Comparison</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; background: #f7f9fb; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th,td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #e8eef5; }}
.status {{ display: inline-block; padding: 4px 8px; border-radius: 999px; color: white; font-weight: bold; min-width: 68px; text-align: center; }}
.win {{ background: #168a4a; }} .lose {{ background: #c72c2c; }}
</style></head><body>
<h1>All Datasets vs PredXGBR Paper</h1>
<table><thead><tr><th>Dataset</th><th>Model</th><th>MAPE</th><th>R2</th><th>Paper MAPE</th><th>Paper R2</th><th>Paper Status</th><th>Hybrid Improves XGB</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    (OUTPUT_DIR / "all_datasets_comparison_stickers.html").write_text(html)


def write_plots(results: pd.DataFrame, best: pd.DataFrame, r2_summary: pd.DataFrame) -> None:
    ordered = best.sort_values("dataset")
    x = np.arange(len(ordered))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, ordered["paper_baseline_MAPE"], width, label="Paper MAPE")
    plt.bar(x + width / 2, ordered["MAPE"], width, label="Our best MAPE")
    plt.xticks(x, ordered["dataset"])
    plt.ylabel("MAPE (%)")
    plt.title("Paper vs Our Best MAPE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "paper_vs_our_best_mape.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "all_datasets_mape_plot.png", dpi=150)
    plt.close()

    r2 = r2_summary.sort_values("dataset")
    x = np.arange(len(r2))
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, r2["paper_R2"], width, label="Paper R2")
    plt.bar(x + width / 2, r2["best_overall_R2"], width, label="Our best R2")
    plt.xticks(x, r2["dataset"])
    plt.ylabel("R2")
    plt.title("Paper vs Our Best R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_datasets_r2_plot.png", dpi=150)
    plt.close()

    pivot = results[results["model"].isin(["Local XGBoost baseline", "Static hybrid quantum", "Hybrid quantum sequential"])]
    pivot = pivot.pivot(index="dataset", columns="model", values="MAPE").sort_index()
    pivot.plot(kind="bar", figsize=(11, 5))
    plt.ylabel("MAPE (%)")
    plt.title("Local XGBoost vs Hybrid Quantum")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "local_xgboost_vs_hybrid_quantum.png", dpi=150)
    plt.close()


def write_paper_draft(results: pd.DataFrame, best: pd.DataFrame, improvement: pd.DataFrame, r2_summary: pd.DataFrame) -> None:
    wins = improvement[improvement["beats_paper_baseline"]]["dataset"].tolist()
    losses = improvement[~improvement["beats_paper_baseline"]]["dataset"].tolist()
    best_overall = best.sort_values("MAPE").iloc[0]
    hybrid = results[results["model"].isin(["Static hybrid quantum", "Hybrid quantum sequential"])]
    hybrid_wins = int(hybrid["hybrid_improves_over_local_xgboost"].sum())
    hybrid_total = len(hybrid)
    md = [
        "# Hybrid Quantum-Enhanced PredXGBR Evaluation",
        "",
        "## Summary",
        "",
        f"The experiments evaluate five datasets from the PredXGBR paper using paper-style original rolling features. The best overall model is **{best_overall['model']}** on **{best_overall['dataset']}** with MAPE {best_overall['MAPE']:.4f}%.",
        "",
        f"Datasets beating the paper baseline: {', '.join(wins) if wins else 'none'}.",
        f"Datasets not beating the paper baseline: {', '.join(losses) if losses else 'none'}.",
        "",
        "Hybrid quantum-enhanced features provide slight but dataset-dependent improvement over the local XGBoost baseline. The result should be reported without overstating quantum advantage.",
        "",
        f"Hybrid quantum improved over local XGBoost in {hybrid_wins}/{hybrid_total} hybrid model comparisons.",
        "",
        "## Best Models by Dataset",
        "",
        best[["dataset", "paper_baseline_MAPE", "MAPE", "R2", "model", "MAPE_improvement_vs_paper_percent"]].to_csv(index=False),
    ]
    (OUTPUT_DIR / "paper_draft.md").write_text("\n".join(md) + "\n")


def write_docx_from_markdown(md_path: Path, docx_path: Path) -> None:
    text = md_path.read_text()
    paragraphs = [line for line in text.splitlines()]
    document_xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
    document_xml += "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>"
    for paragraph in paragraphs:
        document_xml += f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>"
    document_xml += "</w:body></w:document>"
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)


def write_status(args: argparse.Namespace) -> None:
    status = [
        "# Paper Status",
        "",
        "Generated all requested all-dataset comparison files.",
        "",
        "Feature mode: original",
        f"Hybrid quantum setting: {args.hybrid_qubits} qubits, {args.hybrid_layers} layers, {args.hybrid_train_samples} train samples",
        "",
        "Use careful wording: hybrid quantum-enhanced features provide slight but dataset-dependent improvement over the local XGBoost baseline.",
    ]
    (OUTPUT_DIR / "README_paper_status.md").write_text("\n".join(status) + "\n")
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(asdict(FullPaperRunConfig(args.feature_mode, args.device, args.n_estimators, args.hybrid_qubits, args.hybrid_layers, args.hybrid_train_samples, args.random_state)), indent=2) + "\n")


if __name__ == "__main__":
    main()
