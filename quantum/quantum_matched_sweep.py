from __future__ import annotations

import argparse
import csv
import json
import time
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

from baseline_pipeline import DATASETS, RESULTS_DIR, load_dataset, split_features


OUTPUT_DIR = RESULTS_DIR / "quantum_matched_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep GPU quantum feature models against matched classical baselines.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="PJME")
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--split-date", default=None)
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="causal")
    parser.add_argument("--device", default="lightning.gpu")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--qubits", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--log-dir", type=Path, default=Path("logs") / "quantum_matched_sweep")
    return parser.parse_args()


def sample_rows(X: pd.DataFrame, y: pd.Series, samples: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if samples <= 0 or len(X) <= samples:
        return X, y
    index = X.sample(n=samples, random_state=seed).sort_index().index
    return X.loc[index], y.loc[index]


def make_circuit(n_qubits: int, weights: np.ndarray, device_name: str):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x: np.ndarray):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def quantum_transform(X: np.ndarray, circuit) -> np.ndarray:
    return np.asarray([circuit(row) for row in X], dtype=float)


def metrics(model: str, y_true: pd.Series, pred: np.ndarray, seconds: float, extra: dict) -> dict:
    row = {
        "model": model,
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "MAPE": np.mean(np.abs((y_true.to_numpy() - pred) / y_true.to_numpy())) * 100,
        "R2": r2_score(y_true, pred),
        "seconds": seconds,
    }
    row.update(extra)
    return row


def run_single_setting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_qubits: int,
    n_layers: int,
    seed: int,
    device_name: str,
) -> list[dict]:
    preprocessing = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=n_qubits, random_state=seed)),
            ("angle_scale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
        ]
    )
    X_train_angles = preprocessing.fit_transform(X_train)
    X_test_angles = preprocessing.transform(X_test)

    extra = {"n_qubits": n_qubits, "n_layers": n_layers, "seed": seed}
    rows = []

    start = time.perf_counter()
    classical_ridge = Ridge(alpha=1.0)
    classical_ridge.fit(X_train_angles, y_train)
    rows.append(metrics("classical_pca_ridge", y_test, classical_ridge.predict(X_test_angles), time.perf_counter() - start, extra))

    start = time.perf_counter()
    classical_xgb = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.08,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
    )
    classical_xgb.fit(X_train_angles, y_train, verbose=False)
    rows.append(metrics("classical_pca_xgboost", y_test, classical_xgb.predict(X_test_angles), time.perf_counter() - start, extra))

    rng = np.random.default_rng(seed)
    weights = rng.uniform(low=-np.pi, high=np.pi, size=(n_layers, n_qubits, 3))
    circuit = make_circuit(n_qubits, weights, device_name)

    start = time.perf_counter()
    Z_train = quantum_transform(X_train_angles, circuit)
    Z_test = quantum_transform(X_test_angles, circuit)
    quantum_ridge = Ridge(alpha=1.0)
    quantum_ridge.fit(Z_train, y_train)
    rows.append(metrics("quantum_feature_ridge", y_test, quantum_ridge.predict(Z_test), time.perf_counter() - start, extra))

    X_train_hybrid = np.column_stack([X_train_angles, Z_train])
    X_test_hybrid = np.column_stack([X_test_angles, Z_test])

    start = time.perf_counter()
    hybrid_ridge = Ridge(alpha=1.0)
    hybrid_ridge.fit(X_train_hybrid, y_train)
    rows.append(metrics("hybrid_pca_quantum_ridge", y_test, hybrid_ridge.predict(X_test_hybrid), time.perf_counter() - start, extra))

    start = time.perf_counter()
    quantum_xgb = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.08,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
    )
    quantum_xgb.fit(Z_train, y_train, verbose=False)
    rows.append(metrics("quantum_feature_xgboost", y_test, quantum_xgb.predict(Z_test), time.perf_counter() - start, extra))

    start = time.perf_counter()
    hybrid_xgb = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.08,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
    )
    hybrid_xgb.fit(X_train_hybrid, y_train, verbose=False)
    rows.append(metrics("hybrid_pca_quantum_xgboost", y_test, hybrid_xgb.predict(X_test_hybrid), time.perf_counter() - start, extra))

    return rows


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    dataset = args.dataset.upper()
    split_date = args.split_date or DATASETS[dataset]["split_date"]
    df, _, _, _ = load_dataset(dataset, args.data_file)
    X_train_full, y_train_full, X_test_full, y_test_full, _, _ = split_features(df, split_date, args.feature_mode)

    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"quantum_matched_sweep_{run_id}.log"
    partial_path = log_dir / f"quantum_matched_sweep_{run_id}_partial_results.csv"
    config_path = log_dir / f"quantum_matched_sweep_{run_id}_config.json"
    config_path.write_text(json.dumps(serializable_args(args), indent=2) + "\n")
    log_message(log_path, f"Starting matched sweep: dataset={dataset}, train_samples={args.train_samples}, test_samples={args.test_samples}")
    log_message(log_path, f"qubits={args.qubits}, layers={args.layers}, seeds={args.seeds}, device={args.device}")

    all_rows = []
    total_settings = len(args.seeds) * len(args.qubits) * len(args.layers)
    completed_settings = 0
    for seed in args.seeds:
        X_train, y_train = sample_rows(X_train_full, y_train_full, args.train_samples, seed)
        X_test, y_test = sample_rows(X_test_full, y_test_full, args.test_samples, seed)
        for n_qubits in args.qubits:
            for n_layers in args.layers:
                completed_settings += 1
                setting_start = time.perf_counter()
                log_message(
                    log_path,
                    f"[{completed_settings}/{total_settings}] start seed={seed} qubits={n_qubits} layers={n_layers}",
                )
                setting_rows = run_single_setting(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    seed=seed,
                    device_name=args.device,
                )
                all_rows.extend(setting_rows)
                append_partial_results(partial_path, dataset, setting_rows)
                best = min(setting_rows, key=lambda row: row["MAE"])
                log_message(
                    log_path,
                    f"[{completed_settings}/{total_settings}] done seed={seed} qubits={n_qubits} layers={n_layers} "
                    f"seconds={time.perf_counter() - setting_start:.2f} best={best['model']} MAE={best['MAE']:.6f}",
                )

    results = pd.DataFrame(all_rows)
    results.insert(0, "dataset", dataset)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "quantum_matched_sweep_results.csv", index=False)
    winners = summarize_winners(results)
    winners.to_csv(OUTPUT_DIR / "quantum_matched_sweep_winners.csv", index=False)
    (OUTPUT_DIR / "quantum_matched_sweep_config.json").write_text(json.dumps(serializable_args(args), indent=2) + "\n")
    plot_best(results)
    print(results.sort_values("MAE").head(12).to_string(index=False))
    print("\nWinners by setting:")
    print(winners.to_string(index=False))
    log_message(log_path, f"Finished matched sweep. Results saved to {OUTPUT_DIR}")
    log_message(log_path, f"Quantum/hybrid wins: {int(winners['beats_matched_classical'].sum())} of {len(winners)}")
    return results


def log_message(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def append_partial_results(path: Path, dataset: str, rows: list[dict]) -> None:
    fieldnames = ["dataset", "model", "MAE", "RMSE", "MAPE", "R2", "seconds", "n_qubits", "n_layers", "seed"]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"dataset": dataset, **row})


def serializable_args(args: argparse.Namespace) -> dict:
    data = vars(args).copy()
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def summarize_winners(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["seed", "n_qubits", "n_layers"]
    classical = results[results["model"].str.startswith("classical_")]
    best_classical = classical.sort_values(group_cols + ["MAE"]).groupby(group_cols, as_index=False).first()
    best_overall = results.sort_values(group_cols + ["MAE"]).groupby(group_cols, as_index=False).first()
    winners = best_overall.merge(
        best_classical[group_cols + ["model", "MAE"]],
        on=group_cols,
        suffixes=("", "_best_classical"),
    )
    winners["beats_matched_classical"] = winners["MAE"] < winners["MAE_best_classical"]
    winners["MAE_improvement_vs_best_classical"] = winners["MAE_best_classical"] - winners["MAE"]
    return winners


def plot_best(results: pd.DataFrame) -> None:
    best = results.sort_values("MAE").head(20).copy()
    labels = best.apply(lambda row: f"{row['model']} q{row['n_qubits']} l{row['n_layers']} s{row['seed']}", axis=1)
    plt.figure(figsize=(12, 7))
    plt.barh(labels[::-1], best["MAE"].to_numpy()[::-1])
    plt.xlabel("MAE")
    plt.title("Best Matched Classical vs Quantum Runs")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quantum_matched_sweep_best.png", dpi=150)
    plt.close()
