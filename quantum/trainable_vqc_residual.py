from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from baseline_pipeline import DATASETS, MODEL_DIR, RESULTS_DIR, load_dataset, split_features


OUTPUT_DIR = RESULTS_DIR / "trainable_vqc_residual"
MODEL_DIR_OUT = MODEL_DIR / "trainable_vqc_residual"
PAPER_MODEL1 = {
    "PJM": {"MAPE": 1.07, "R2": 0.99},
    "PJME": {"MAPE": 1.28, "R2": 0.99},
    "PJMW": {"MAPE": 1.07, "R2": 0.98},
    "AEP": {"MAPE": 0.98, "R2": 0.99},
    "DAYTON": {"MAPE": 1.12, "R2": 0.99},
}


@dataclass
class TrainableVQCConfig:
    dataset: str
    feature_mode: str
    device: str
    n_qubits: int
    n_layers: int
    max_train_samples: int
    max_test_samples: int
    epochs: int
    learning_rate: float
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VQC residual regressor and compare with PredXGBR/QRC-GBF.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PJME")
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="original")
    parser.add_argument("--device", default="lightning.qubit")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-test-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def metric_row(dataset: str, model: str, y_true: np.ndarray, pred: np.ndarray, seconds: float, notes: str) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "MAPE": float(np.mean(np.abs((y_true - pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, pred)),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "seconds": float(seconds),
        "notes": notes,
    }


def sample_positions(length: int, max_samples: int, random_state: int) -> np.ndarray:
    if max_samples <= 0 or length <= max_samples:
        return np.arange(length)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(length, size=max_samples, replace=False))


def make_qnode(n_qubits: int, device: str):
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def train_vqc(
    X_train_angles: np.ndarray,
    residual_train_scaled: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(args.random_state)
    weights = qml.numpy.array(
        rng.normal(scale=0.1, size=(args.n_layers, args.n_qubits, 3)),
        requires_grad=True,
    )
    bias = qml.numpy.array(0.0, requires_grad=True)
    readout = qml.numpy.array(rng.normal(scale=0.1, size=args.n_qubits), requires_grad=True)
    circuit = make_qnode(args.n_qubits, args.device)
    opt = qml.AdamOptimizer(stepsize=args.learning_rate)
    history = []

    X = qml.numpy.array(X_train_angles, requires_grad=False)
    y = qml.numpy.array(residual_train_scaled, requires_grad=False)

    def predict_one(x, w, r, b):
        z = qml.numpy.stack(circuit(x, w))
        return qml.numpy.dot(z, r) + b

    def cost(w, r, b):
        preds = [predict_one(x, w, r, b) for x in X]
        pred_array = qml.numpy.stack(preds)
        return qml.numpy.mean((pred_array - y) ** 2)

    for epoch in range(1, args.epochs + 1):
        weights, readout, bias, loss = opt.step_and_cost(cost, weights, readout, bias)
        history.append({"epoch": epoch, "loss": float(loss)})
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 5) == 0:
            print(f"epoch={epoch} loss={float(loss):.6f}", flush=True)

    params = {
        "weights": np.asarray(weights, dtype=float),
        "readout": np.asarray(readout, dtype=float),
        "bias": float(bias),
    }
    return params, circuit, history


def predict_vqc(X_angles: np.ndarray, circuit, params: dict) -> np.ndarray:
    preds = []
    weights = params["weights"]
    readout = params["readout"]
    bias = params["bias"]
    for x in X_angles:
        z = np.asarray(circuit(x, weights), dtype=float)
        preds.append(float(np.dot(z, readout) + bias))
    return np.asarray(preds)


def train_backbone(dataset: str, args: argparse.Namespace):
    df, _, _, _ = load_dataset(dataset)
    split_date = DATASETS[dataset]["split_date"]
    X_train, y_train, X_test, y_test, _, test = split_features(df, split_date, args.feature_mode)
    model = xgb.XGBRegressor(
        n_estimators=1000,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=args.random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    seconds = time.perf_counter() - start
    return model, X_train, y_train, X_test, y_test, test, seconds


def run_trainable_vqc_residual(args: argparse.Namespace) -> pd.DataFrame:
    dataset = args.dataset.upper()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Training Local GBF backbone for {dataset}", flush=True)
    backbone, X_train, y_train, X_test, y_test, test, backbone_seconds = train_backbone(dataset, args)
    train_pred = backbone.predict(X_train)
    test_pred_full = backbone.predict(X_test)
    residual = y_train.to_numpy() - train_pred

    train_pos = sample_positions(len(X_train), args.max_train_samples, args.random_state)
    test_pos = sample_positions(len(X_test), args.max_test_samples, args.random_state)
    X_vqc_train = X_train.iloc[train_pos]
    residual_vqc = residual[train_pos]
    X_eval = X_test.iloc[test_pos]
    y_eval = y_test.to_numpy()[test_pos]
    test_pred = test_pred_full[test_pos]

    preprocessing = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=args.n_qubits, random_state=args.random_state)),
            ("angle_scale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
        ]
    )
    X_train_angles = preprocessing.fit_transform(X_vqc_train)
    X_eval_angles = preprocessing.transform(X_eval)
    residual_scaler = StandardScaler()
    residual_scaled = residual_scaler.fit_transform(residual_vqc.reshape(-1, 1)).ravel()

    print(
        f"Training VQC residual model: dataset={dataset}, train={len(X_train_angles)}, test={len(X_eval_angles)}, "
        f"qubits={args.n_qubits}, layers={args.n_layers}, epochs={args.epochs}",
        flush=True,
    )
    start = time.perf_counter()
    params, circuit, history = train_vqc(X_train_angles, residual_scaled, args)
    train_seconds = time.perf_counter() - start
    residual_pred_scaled = predict_vqc(X_eval_angles, circuit, params)
    residual_pred = residual_scaler.inverse_transform(residual_pred_scaled.reshape(-1, 1)).ravel()
    trainable_vqc_pred = test_pred + residual_pred

    paper = PAPER_MODEL1[dataset]
    paper_pred_constant = np.full_like(y_eval, fill_value=np.nan, dtype=float)
    rows = [
        {
            "dataset": dataset,
            "model": "Published PredXGBR-1",
            "MAPE": paper["MAPE"],
            "R2": paper["R2"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "seconds": np.nan,
            "notes": "published Table 3 external baseline",
        },
        metric_row(dataset, "Local GBF-1", y_eval, test_pred, backbone_seconds, "local gradient-boosted backbone on same sampled test rows"),
        metric_row(
            dataset,
            "Trainable VQC Residual QRC-GBF-1",
            y_eval,
            trainable_vqc_pred,
            train_seconds,
            "VQC weights and readout trained directly on GBF residual target",
        ),
    ]
    results = pd.DataFrame(rows)
    results["beats_published_predxgbr"] = results["MAPE"] < paper["MAPE"]
    local_mape = float(results.loc[results["model"] == "Local GBF-1", "MAPE"].iloc[0])
    results["beats_local_gbf"] = results["MAPE"] < local_mape
    results["MAPE_improvement_vs_published_percent"] = (paper["MAPE"] - results["MAPE"]) / paper["MAPE"] * 100
    results["MAPE_improvement_vs_local_gbf_percent"] = (local_mape - results["MAPE"]) / local_mape * 100

    suffix = f"{dataset.lower()}_q{args.n_qubits}_l{args.n_layers}_n{len(X_train_angles)}_e{args.epochs}"
    results.to_csv(OUTPUT_DIR / f"trainable_vqc_residual_{suffix}_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / f"trainable_vqc_residual_{suffix}_history.csv", index=False)
    pd.DataFrame(
        {
            "datetime": test.index[test_pos],
            "actual_load": y_eval,
            "local_gbf_prediction": test_pred,
            "trainable_vqc_prediction": trainable_vqc_pred,
            "trainable_vqc_residual": residual_pred,
        }
    ).to_csv(OUTPUT_DIR / f"trainable_vqc_residual_{suffix}_predictions.csv", index=False)
    with (MODEL_DIR_OUT / f"trainable_vqc_residual_{suffix}.pkl").open("wb") as handle:
        pickle.dump({"params": params, "preprocessing": preprocessing, "residual_scaler": residual_scaler, "config": vars(args)}, handle)
    write_plots(results, history, y_eval, test_pred, trainable_vqc_pred, suffix)
    (OUTPUT_DIR / f"trainable_vqc_residual_{suffix}_config.json").write_text(json.dumps(asdict(config_from_args(args)), indent=2) + "\n")
    print(results.to_string(index=False))
    return results


def config_from_args(args: argparse.Namespace) -> TrainableVQCConfig:
    return TrainableVQCConfig(
        dataset=args.dataset.upper(),
        feature_mode=args.feature_mode,
        device=args.device,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )


def write_plots(results: pd.DataFrame, history: list[dict], actual: np.ndarray, local_pred: np.ndarray, vqc_pred: np.ndarray, suffix: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(results["model"], results["MAPE"], color=["#8d99ae", "#457b9d", "#2a9d8f"])
    plt.ylabel("MAPE (%) lower is better")
    plt.xticks(rotation=20, ha="right")
    plt.title("Trainable VQC Residual Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"trainable_vqc_residual_{suffix}_mape.png", dpi=160)
    plt.close()

    history_df = pd.DataFrame(history)
    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["loss"], marker="o", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Residual MSE loss")
    plt.title("Trainable VQC Residual Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"trainable_vqc_residual_{suffix}_loss.png", dpi=160)
    plt.close()

    view = min(500, len(actual))
    plt.figure(figsize=(12, 4))
    plt.plot(actual[:view], label="Actual", linewidth=1)
    plt.plot(local_pred[:view], label="Local GBF-1", linewidth=1)
    plt.plot(vqc_pred[:view], label="Trainable VQC residual", linewidth=1)
    plt.legend()
    plt.title("Trainable VQC Residual Forecast Sample")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"trainable_vqc_residual_{suffix}_forecast.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    run_trainable_vqc_residual(parse_args())
