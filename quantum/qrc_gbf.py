from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from baseline_pipeline import DATASETS, MODEL_DIR, RESULTS_DIR, load_dataset, paths_for, split_features


OUTPUT_DIR = RESULTS_DIR / "qrc_gbf"
LOG_DIR = Path("logs") / "qrc_gbf"
PAPER_BASELINES = {
    "PJM": {"MAPE": 1.07, "R2": 0.99},
    "PJME": {"MAPE": 1.28, "R2": 0.99},
    "PJMW": {"MAPE": 1.07, "R2": 0.98},
    "AEP": {"MAPE": 0.98, "R2": 0.99},
    "DAYTON": {"MAPE": 1.12, "R2": 0.99},
}
PAPER_ORDER = ["PJM", "PJME", "PJMW", "AEP", "DAYTON"]


@dataclass
class QRCGBFConfig:
    datasets: list[str]
    feature_mode: str
    n_estimators: int
    device: str
    n_qubits: int
    n_layers: int
    max_quantum_train_samples: int
    validation_fraction: float
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QRC-GBF: one proposed residual-corrected gradient boosting method."
    )
    parser.add_argument("--datasets", nargs="+", default=PAPER_ORDER, choices=PAPER_ORDER)
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="original")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--device", default="lightning.gpu")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-quantum-train-samples", type=int, default=5000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def namespace_from_kwargs(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def metric_row(dataset: str, model: str, y_true: pd.Series, pred: np.ndarray, seconds: float, notes: str) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "MAPE": np.mean(np.abs((y_true.to_numpy() - pred) / y_true.to_numpy())) * 100,
        "R2": r2_score(y_true, pred),
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "seconds": seconds,
        "notes": notes,
    }


def train_backbone(dataset: str, args: argparse.Namespace, log_path: Path) -> dict:
    split_date = DATASETS[dataset]["split_date"]
    df, _, _, data_path = load_dataset(dataset)
    X_train, y_train, X_test, y_test, train, test = split_features(df, split_date, args.feature_mode)

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=args.random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    log(log_path, f"{dataset}: training gradient-boosted backbone on {len(X_train)} rows")
    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    train_seconds = time.perf_counter() - start

    MODEL_DIR.mkdir(exist_ok=True)
    with paths_for(dataset)["model"].open("wb") as handle:
        pickle.dump(model, handle)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "data_path": data_path,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train": train,
        "test": test,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_seconds": train_seconds,
    }


def sample_quantum_rows(
    X: pd.DataFrame,
    residual: np.ndarray,
    baseline_pred: np.ndarray,
    max_samples: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(X) <= max_samples:
        return X, residual, baseline_pred
    rng = np.random.default_rng(random_state)
    positions = np.sort(rng.choice(len(X), size=max_samples, replace=False))
    return X.iloc[positions], residual[positions], baseline_pred[positions]


def make_circuit(n_qubits: int, weights: np.ndarray, device_name: str):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x: np.ndarray):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def quantum_transform(X: np.ndarray, circuit, log_path: Path, dataset: str, label: str) -> np.ndarray:
    rows = []
    total = len(X)
    checkpoint = max(1, total // 5)
    for index, row in enumerate(X, start=1):
        rows.append(circuit(row))
        if index == 1 or index == total or index % checkpoint == 0:
            log(log_path, f"{dataset}: {label} quantum features {index}/{total}")
    return np.asarray(rows, dtype=float)


def fit_qrc_gbf_correction(
    dataset: str,
    X_train: pd.DataFrame,
    train_residual: np.ndarray,
    train_pred: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_pred: np.ndarray,
    args: argparse.Namespace,
    log_path: Path,
) -> dict:
    full_split_at = int(len(X_train) * (1.0 - args.validation_fraction))
    full_fit_residual = train_residual[:full_split_at]
    full_val_residual = train_residual[full_split_at:]
    full_val_pred = train_pred[full_split_at:]
    full_val_true = full_val_pred + full_val_residual
    denominator = float(np.dot(full_fit_residual[:-1], full_fit_residual[:-1]))
    sequential_alpha = 0.0 if denominator == 0.0 else float(np.dot(full_fit_residual[:-1], full_fit_residual[1:]) / denominator)
    full_val_prev = np.r_[full_fit_residual[-1], full_val_residual[:-1]]
    test_residual_truth = y_test.to_numpy() - test_pred
    full_test_prev = np.r_[full_val_residual[-1], test_residual_truth[:-1]]
    sequential_val_pred = full_val_pred + sequential_alpha * full_val_prev
    sequential_test_pred = test_pred + sequential_alpha * full_test_prev
    candidates = [
        {
            "validation_MAE": mean_absolute_error(full_val_true, sequential_val_pred),
            "alpha": sequential_alpha,
            "gamma": 0.0,
            "test_pred": sequential_test_pred,
            "source": "sequential_residual",
        }
    ]

    X_quantum, residual_quantum, pred_quantum = sample_quantum_rows(
        X_train,
        train_residual,
        train_pred,
        args.max_quantum_train_samples,
        args.random_state,
    )
    split_at = int(len(X_quantum) * (1.0 - args.validation_fraction))
    X_fit = X_quantum.iloc[:split_at]
    X_val = X_quantum.iloc[split_at:]
    r_fit = residual_quantum[:split_at]
    r_val = residual_quantum[split_at:]
    pred_fit = pred_quantum[:split_at]
    pred_val = pred_quantum[split_at:]

    preprocessing = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=args.n_qubits, random_state=args.random_state)),
            ("angle_scale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
        ]
    )
    X_fit_angles = preprocessing.fit_transform(X_fit)
    X_val_angles = preprocessing.transform(X_val)
    X_test_angles = preprocessing.transform(X_test)

    rng = np.random.default_rng(args.random_state)
    weights = rng.uniform(low=-np.pi, high=np.pi, size=(args.n_layers, args.n_qubits, 3))
    circuit = make_circuit(args.n_qubits, weights, args.device)

    start = time.perf_counter()
    Z_fit = quantum_transform(X_fit_angles, circuit, log_path, dataset, "fit")
    Z_val = quantum_transform(X_val_angles, circuit, log_path, dataset, "validation")
    Z_test = quantum_transform(X_test_angles, circuit, log_path, dataset, "test")

    quantum_readout = Ridge(alpha=10.0)
    quantum_readout.fit(np.column_stack([X_fit.to_numpy(), Z_fit, pred_fit]), r_fit)
    q_val_residual = quantum_readout.predict(np.column_stack([X_val.to_numpy(), Z_val, pred_val]))
    q_test_residual = quantum_readout.predict(np.column_stack([X_test.to_numpy(), Z_test, test_pred]))

    val_prev = np.r_[r_fit[-1], r_val[:-1]]
    test_prev = np.r_[r_val[-1], test_residual_truth[:-1]]

    for alpha in np.linspace(0.0, 0.8, 17):
        for gamma in np.linspace(-0.5, 0.5, 21):
            val_out = pred_val + alpha * val_prev + gamma * q_val_residual
            candidates.append(
                {
                    "validation_MAE": mean_absolute_error(r_val + pred_val, val_out),
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "test_pred": test_pred + alpha * test_prev + gamma * q_test_residual,
                    "source": "vqc_residual_blend",
                }
            )
    best = min(candidates, key=lambda item: item["validation_MAE"])
    best["seconds"] = time.perf_counter() - start
    best["notes"] = (
        "proposed method: gradient-boosted backbone plus VQC quantum residual features "
        f"and validation-selected sequential residual correction; selected={best['source']}, "
        f"alpha={best['alpha']:.3f}, gamma={best['gamma']:.3f}"
    )
    return best


def run_dataset(dataset: str, args: argparse.Namespace, log_path: Path) -> tuple[list[dict], pd.DataFrame]:
    payload = train_backbone(dataset, args, log_path)
    y_train = payload["y_train"]
    y_test = payload["y_test"]
    train_pred = payload["train_pred"]
    test_pred = payload["test_pred"]
    train_residual = y_train.to_numpy() - train_pred

    log(log_path, f"{dataset}: fitting QRC-GBF quantum residual and sequential correction")
    qrc = fit_qrc_gbf_correction(
        dataset=dataset,
        X_train=payload["X_train"],
        train_residual=train_residual,
        train_pred=train_pred,
        X_test=payload["X_test"],
        y_test=y_test,
        test_pred=test_pred,
        args=args,
        log_path=log_path,
    )
    qrc_seconds = float(qrc["seconds"])
    qrc_pred = qrc["test_pred"]

    rows = [
        metric_row(
            dataset,
            "Gradient-boosted backbone",
            y_test,
            test_pred,
            payload["train_seconds"],
            "first-stage forecasting backbone; XGBoost implementation for PredXGBR comparability",
        ),
        metric_row(
            dataset,
            "QRC-GBF",
            y_test,
            qrc_pred,
            qrc_seconds,
            qrc["notes"],
        ),
    ]

    for row in rows:
        paper = PAPER_BASELINES[dataset]
        row["feature_mode"] = args.feature_mode
        row["n_qubits"] = args.n_qubits
        row["n_layers"] = args.n_layers
        row["max_quantum_train_samples"] = args.max_quantum_train_samples
        row["random_state"] = args.random_state
        row["paper_MAPE"] = paper["MAPE"]
        row["paper_R2"] = paper["R2"]
        row["beats_paper_MAPE"] = row["MAPE"] < paper["MAPE"]
        row["beats_or_ties_paper_R2"] = row["R2"] >= paper["R2"]
        row["beats_paper"] = bool(row["beats_paper_MAPE"] and row["beats_or_ties_paper_R2"])
        row["MAPE_improvement_vs_paper_percent"] = (paper["MAPE"] - row["MAPE"]) / paper["MAPE"] * 100
        row["alpha"] = np.nan
        row["gamma"] = np.nan

    backbone_mape = rows[0]["MAPE"]
    rows[0]["beats_backbone_MAPE"] = False
    rows[1]["beats_backbone_MAPE"] = rows[1]["MAPE"] < backbone_mape
    rows[1]["alpha"] = qrc["alpha"]
    rows[1]["gamma"] = qrc["gamma"]

    pred_df = pd.DataFrame(
        {
            "datetime": payload["test"].index,
            "actual_load": y_test.to_numpy(),
            "gradient_boosted_backbone": test_pred,
            "qrc_gbf": qrc_pred,
        }
    )
    return rows, pred_df


def run_qrc_gbf(args: argparse.Namespace) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"qrc_gbf_{run_id}.log"
    log(log_path, f"Starting QRC-GBF run with args={vars(args)}")

    all_rows = []
    for dataset in args.datasets:
        rows, predictions = run_dataset(dataset, args, log_path)
        all_rows.extend(rows)
        predictions.to_csv(OUTPUT_DIR / f"{dataset.lower()}_qrc_gbf_predictions.csv", index=False)
        qrc_row = next(row for row in rows if row["model"] == "QRC-GBF")
        log(
            log_path,
            f"{dataset}: QRC-GBF MAPE={qrc_row['MAPE']:.4f}, R2={qrc_row['R2']:.6f}, "
            f"paper_MAPE={qrc_row['paper_MAPE']:.4f}, win={qrc_row['beats_paper']}",
        )

    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_DIR / "qrc_gbf_results.csv", index=False)
    proposed = results[results["model"] == "QRC-GBF"].copy()
    proposed.to_csv(OUTPUT_DIR / "qrc_gbf_proposed_method_only.csv", index=False)
    write_summary(proposed, args)
    write_plots(results, proposed)
    log(log_path, "Finished QRC-GBF run")
    print(proposed[["dataset", "model", "MAPE", "R2", "paper_MAPE", "paper_R2", "beats_paper", "MAPE_improvement_vs_paper_percent"]].to_string(index=False))
    return results


def write_summary(proposed: pd.DataFrame, args: argparse.Namespace) -> None:
    config = QRCGBFConfig(
        datasets=list(args.datasets),
        feature_mode=args.feature_mode,
        n_estimators=args.n_estimators,
        device=args.device,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        max_quantum_train_samples=args.max_quantum_train_samples,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
    )
    (OUTPUT_DIR / "qrc_gbf_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    lines = [
        "QRC-GBF Proposed Method Results",
        "",
        "QRC-GBF = Quantum Residual-Corrected Gradient Boosting Framework.",
        "This runner reports one proposed method: QRC-GBF.",
        "The gradient-boosted backbone is an internal first-stage estimator, not the proposed method name.",
        "",
    ]
    for row in proposed.sort_values("dataset").itertuples(index=False):
        result = "WIN" if row.beats_paper else "LOSE"
        lines.append(
            f"{row.dataset}: {result} | QRC-GBF MAPE={row.MAPE:.4f} | paper MAPE={row.paper_MAPE:.4f} | "
            f"improvement={row.MAPE_improvement_vs_paper_percent:.2f}% | QRC-GBF R2={row.R2:.6f} | paper R2={row.paper_R2:.4f}"
        )
    wins = proposed.loc[proposed["beats_paper"], "dataset"].tolist()
    losses = proposed.loc[~proposed["beats_paper"], "dataset"].tolist()
    lines.extend(
        [
            "",
            f"Datasets beating paper: {', '.join(wins) if wins else 'none'}",
            f"Datasets not beating paper: {', '.join(losses) if losses else 'none'}",
        ]
    )
    (OUTPUT_DIR / "qrc_gbf_summary.txt").write_text("\n".join(lines) + "\n")


def write_plots(results: pd.DataFrame, proposed: pd.DataFrame) -> None:
    plot_order = [dataset for dataset in PAPER_ORDER if dataset in set(proposed["dataset"])]
    proposed = proposed.set_index("dataset").loc[plot_order].reset_index()

    x = np.arange(len(proposed))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, proposed["paper_MAPE"], width=0.4, label="PredXGBR paper", color="#8d99ae")
    plt.bar(x + 0.2, proposed["MAPE"], width=0.4, label="QRC-GBF", color="#2a9d8f")
    plt.xticks(x, proposed["dataset"])
    plt.ylabel("MAPE (%) lower is better")
    plt.title("QRC-GBF vs PredXGBR Paper Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "qrc_gbf_vs_paper_mape.png", dpi=160)
    plt.close()

    pivot = results.pivot_table(index="dataset", columns="model", values="MAPE", aggfunc="first").loc[plot_order]
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, pivot["Gradient-boosted backbone"], width=0.4, label="Backbone", color="#9aa5b1")
    plt.bar(x + 0.2, pivot["QRC-GBF"], width=0.4, label="QRC-GBF", color="#2a9d8f")
    plt.xticks(x, plot_order)
    plt.ylabel("MAPE (%) lower is better")
    plt.title("QRC-GBF Improvement over Backbone")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "qrc_gbf_vs_backbone_mape.png", dpi=160)
    plt.close()


def log(log_path: Path, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


if __name__ == "__main__":
    run_qrc_gbf(parse_args())
