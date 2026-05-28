from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum.qrc_gbf import LOG_DIR, OUTPUT_DIR, PAPER_ORDER, run_dataset


SWEEP_DIR = OUTPUT_DIR / "sweep"


@dataclass
class SweepConfig:
    datasets: list[str]
    feature_mode: str
    n_estimators: int
    device: str
    qubits: list[int]
    layers: list[int]
    seeds: list[int]
    max_quantum_train_samples: int
    validation_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep QRC-GBF VQC settings across paper datasets.")
    parser.add_argument("--datasets", nargs="+", default=PAPER_ORDER, choices=PAPER_ORDER)
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="original")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--device", default="lightning.gpu")
    parser.add_argument("--qubits", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max-quantum-train-samples", type=int, default=5000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"qrc_gbf_sweep_{run_id}.log"
    log(log_path, f"Starting QRC-GBF sweep with args={vars(args)}")

    all_rows = []
    total = len(args.datasets) * len(args.qubits) * len(args.layers) * len(args.seeds)
    counter = 0
    for seed in args.seeds:
        for n_qubits in args.qubits:
            for n_layers in args.layers:
                setting_args = argparse.Namespace(
                    datasets=args.datasets,
                    feature_mode=args.feature_mode,
                    n_estimators=args.n_estimators,
                    device=args.device,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    max_quantum_train_samples=args.max_quantum_train_samples,
                    validation_fraction=args.validation_fraction,
                    random_state=seed,
                )
                for dataset in args.datasets:
                    counter += 1
                    start = time.perf_counter()
                    log(
                        log_path,
                        f"[{counter}/{total}] dataset={dataset} qubits={n_qubits} layers={n_layers} seed={seed}",
                    )
                    rows, _ = run_dataset(dataset, setting_args, log_path)
                    for row in rows:
                        row["sweep_seconds"] = time.perf_counter() - start
                    all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    proposed = results[results["model"] == "QRC-GBF"].copy()
    results.to_csv(SWEEP_DIR / "qrc_gbf_sweep_all_rows.csv", index=False)
    proposed.to_csv(SWEEP_DIR / "qrc_gbf_sweep_proposed_only.csv", index=False)

    best_by_dataset = proposed.sort_values(["dataset", "MAPE"]).groupby("dataset", as_index=False).first()
    best_by_dataset.to_csv(SWEEP_DIR / "qrc_gbf_sweep_best_by_dataset.csv", index=False)
    write_summary(proposed, best_by_dataset)
    write_plots(proposed, best_by_dataset)

    config = SweepConfig(
        datasets=list(args.datasets),
        feature_mode=args.feature_mode,
        n_estimators=args.n_estimators,
        device=args.device,
        qubits=list(args.qubits),
        layers=list(args.layers),
        seeds=list(args.seeds),
        max_quantum_train_samples=args.max_quantum_train_samples,
        validation_fraction=args.validation_fraction,
    )
    (SWEEP_DIR / "qrc_gbf_sweep_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    log(log_path, "Finished QRC-GBF sweep")
    print(best_by_dataset[["dataset", "MAPE", "R2", "n_qubits", "n_layers", "random_state", "paper_MAPE", "beats_paper", "beats_backbone_MAPE"]].to_string(index=False))


def write_summary(proposed: pd.DataFrame, best: pd.DataFrame) -> None:
    total = len(proposed)
    paper_wins = int(proposed["beats_paper"].sum())
    backbone_wins = int(proposed["beats_backbone_MAPE"].sum())
    lines = [
        "QRC-GBF VQC Sweep Proof Summary",
        "",
        f"Total QRC-GBF settings tested: {total}",
        f"Settings beating PredXGBR paper baseline: {paper_wins}/{total}",
        f"Settings beating local gradient-boosted backbone: {backbone_wins}/{total}",
        "",
        "Best QRC-GBF setting by dataset:",
    ]
    for row in best.sort_values("dataset").itertuples(index=False):
        lines.append(
            f"{row.dataset}: MAPE={row.MAPE:.4f}, R2={row.R2:.6f}, "
            f"q={row.n_qubits}, layers={row.n_layers}, seed={row.random_state}, "
            f"paper_win={row.beats_paper}, backbone_win={row.beats_backbone_MAPE}, "
            f"paper_improvement={row.MAPE_improvement_vs_paper_percent:.2f}%"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "The sweep proves whether the QRC-GBF VQC branch is robust across configurations.",
            "A paper claim should separate paper-baseline wins from backbone-improvement wins.",
        ]
    )
    (SWEEP_DIR / "qrc_gbf_sweep_summary.txt").write_text("\n".join(lines) + "\n")


def write_plots(proposed: pd.DataFrame, best: pd.DataFrame) -> None:
    labels = proposed.apply(lambda row: f"{row['dataset']} q{int(row['n_qubits'])} l{int(row['n_layers'])} s{int(row['random_state'])}", axis=1)
    ordered = proposed.assign(label=labels).sort_values("MAPE")
    plt.figure(figsize=(13, 7))
    colors = ["#2a9d8f" if value else "#9aa5b1" for value in ordered["beats_backbone_MAPE"]]
    plt.barh(ordered["label"][::-1], ordered["MAPE"].to_numpy()[::-1], color=colors[::-1])
    plt.xlabel("MAPE (%) lower is better")
    plt.title("QRC-GBF VQC Sweep")
    plt.tight_layout()
    plt.savefig(SWEEP_DIR / "qrc_gbf_sweep_mape_rank.png", dpi=160)
    plt.close()

    best = best.sort_values("dataset")
    plt.figure(figsize=(10, 5))
    plt.bar(best["dataset"], best["MAPE"], color="#2a9d8f")
    plt.plot(best["dataset"], best["paper_MAPE"], color="#d62828", marker="o", label="PredXGBR paper MAPE")
    plt.ylabel("MAPE (%) lower is better")
    plt.title("Best QRC-GBF Sweep Result vs Paper")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SWEEP_DIR / "qrc_gbf_sweep_best_vs_paper.png", dpi=160)
    plt.close()


def log(log_path: Path, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    main()
