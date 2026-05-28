from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_RUNS_DIR = ROOT / "paper_runs"
PYTHON = ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and preserve paper-ready PredXGBR hybrid quantum results.")
    parser.add_argument("--dataset", default="PJME")
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="causal")
    parser.add_argument("--skip-baseline", action="store_true", help="Reuse the existing trained XGBoost baseline.")
    parser.add_argument("--skip-matched-sweep", action="store_true", help="Skip the long matched low-data quantum sweep.")
    parser.add_argument("--matched-train-samples", type=int, default=256)
    parser.add_argument("--matched-test-samples", type=int, default=256)
    parser.add_argument("--matched-qubits", nargs="+", default=["2", "3", "4", "6"])
    parser.add_argument("--matched-layers", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--matched-seeds", nargs="+", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--hybrid-qubits", type=int, default=4)
    parser.add_argument("--hybrid-layers", type=int, default=2)
    parser.add_argument("--hybrid-train-samples", type=int, default=5000)
    parser.add_argument("--device", default="lightning.gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PAPER_RUNS_DIR / run_id
    log_dir = run_dir / "logs"
    artifact_dir = run_dir / "artifacts"
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "args": serializable_args(args),
        "commands": [],
    }

    if not args.skip_baseline:
        run_command(
            [
                str(PYTHON),
                "run_all.py",
                "--dataset",
                args.dataset,
                "--feature-mode",
                args.feature_mode,
            ],
            log_dir / "01_xgboost_baseline.log",
            manifest,
        )

    run_command(
        [
            str(PYTHON),
            "run_postprocess_baseline_sweep.py",
            "--dataset",
            args.dataset,
            "--feature-mode",
            args.feature_mode,
        ],
        log_dir / "02_postprocess_baseline_sweep.log",
        manifest,
    )

    run_command(
        [
            str(PYTHON),
            "run_hybrid_quantum_postprocess.py",
            "--dataset",
            args.dataset,
            "--feature-mode",
            args.feature_mode,
            "--device",
            args.device,
            "--n-qubits",
            str(args.hybrid_qubits),
            "--n-layers",
            str(args.hybrid_layers),
            "--max-train-samples",
            str(args.hybrid_train_samples),
            "--log-dir",
            str(log_dir / "hybrid_quantum_postprocess"),
        ],
        log_dir / "03_hybrid_quantum_postprocess.log",
        manifest,
    )

    if not args.skip_matched_sweep:
        run_command(
            [
                str(PYTHON),
                "run_quantum_matched_sweep.py",
                "--dataset",
                args.dataset,
                "--feature-mode",
                args.feature_mode,
                "--device",
                args.device,
                "--train-samples",
                str(args.matched_train_samples),
                "--test-samples",
                str(args.matched_test_samples),
                "--qubits",
                *args.matched_qubits,
                "--layers",
                *args.matched_layers,
                "--seeds",
                *args.matched_seeds,
                "--log-dir",
                str(log_dir / "quantum_matched_sweep"),
            ],
            log_dir / "04_quantum_matched_sweep.log",
            manifest,
        )
        run_command([str(PYTHON), "run_quantum_matched_plots.py"], log_dir / "05_quantum_matched_plots.log", manifest)

    run_command([str(PYTHON), "run_quantum_all_comparison.py"], log_dir / "06_all_comparison.log", manifest)

    copy_artifacts(artifact_dir)
    build_paper_summary(artifact_dir)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Paper run saved to {run_dir}")


def run_command(command: list[str], log_path: Path, manifest: dict) -> None:
    manifest["commands"].append({"command": command, "log": str(log_path.relative_to(ROOT))})
    print(f"Running: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def copy_artifacts(artifact_dir: Path) -> None:
    paths = [
        ROOT / "results" / "metrics.csv",
        ROOT / "results" / "actual_vs_predicted.png",
        ROOT / "results" / "postprocess_baseline_sweep",
        ROOT / "results" / "hybrid_quantum_postprocess",
        ROOT / "results" / "quantum_matched_sweep",
        ROOT / "results" / "quantum_comparison",
    ]
    for path in paths:
        if not path.exists():
            continue
        destination = artifact_dir / path.relative_to(ROOT / "results")
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def build_paper_summary(artifact_dir: Path) -> None:
    comparison_path = artifact_dir / "quantum_comparison" / "all_quantum_experiment_comparison.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    summary_rows = []
    for scope, group in comparison.groupby("comparison_scope"):
        best = group.sort_values("MAE").iloc[0]
        summary_rows.append(
            {
                "comparison_scope": scope,
                "best_model": best["model"],
                "best_MAE": best["MAE"],
                "best_RMSE": best["RMSE"],
                "best_MAPE": best["MAPE"],
                "best_R2": best["R2"],
                "notes": best["notes"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("best_MAE")
    summary.to_csv(artifact_dir / "paper_summary_best_by_scope.csv", index=False)
    (artifact_dir / "paper_summary_best_by_scope.md").write_text(dataframe_to_markdown(summary) + "\n")


def serializable_args(args: argparse.Namespace) -> dict:
    return vars(args).copy()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False):
        values = [str(value).replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
