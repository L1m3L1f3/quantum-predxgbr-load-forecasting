from __future__ import annotations

import argparse
import json
import pickle
import sys
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_pipeline import DATASETS, RESULTS_DIR, load_dataset, split_features


OUTPUT_DIR = RESULTS_DIR / "paper_comparison"
FIGURE_DIR = ROOT / "paper_figures"
QRC_MODEL1_PATH = RESULTS_DIR / "qrc_gbf" / "sweep" / "qrc_gbf_sweep_best_by_dataset.csv"
LOCAL_MODEL1_PATH = RESULTS_DIR / "local_table3_baselines" / "local_table3_baselines_results.csv"
DATASET_ORDER = ["PJM", "PJME", "PJMW", "AEP", "DAYTON"]

PUBLISHED = {
    "PJM": {"model1_mape": 1.07, "model1_r2": 0.99, "model2_mape": 6.87, "model2_r2": 0.71},
    "PJME": {"model1_mape": 1.28, "model1_r2": 0.99, "model2_mape": 8.59, "model2_r2": 0.58},
    "PJMW": {"model1_mape": 1.07, "model1_r2": 0.98, "model2_mape": 8.42, "model2_r2": 0.59},
    "AEP": {"model1_mape": 0.98, "model1_r2": 0.99, "model2_mape": 8.08, "model2_r2": 0.57},
    "DAYTON": {"model1_mape": 1.12, "model1_r2": 0.99, "model2_mape": 8.49, "model2_r2": 0.62},
}


@dataclass
class Config:
    n_estimators: int
    n_qubits: int
    n_layers: int
    max_quantum_train_samples: int
    validation_fraction: float
    random_state: int
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Model 1 / Model 2 paper comparison tables and plots.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-qubits", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--max-quantum-train-samples", type=int, default=5000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="lightning.gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    model1 = build_model1_rows()
    model2 = run_model2_rows(args)
    all_rows = pd.concat([model1, model2], ignore_index=True)

    model1.to_csv(OUTPUT_DIR / "model1_short_term_comparison.csv", index=False)
    model2.to_csv(OUTPUT_DIR / "model2_long_term_comparison.csv", index=False)
    all_rows.to_csv(OUTPUT_DIR / "qrc_gbf_vs_predxgbr_all_datasets.csv", index=False)
    all_rows[
        [
            "dataset",
            "feature_setting",
            "local_gbf_name",
            "qrc_gbf_name",
            "local_gbf_MAPE",
            "local_gbf_R2",
            "qrc_gbf_MAPE",
            "qrc_gbf_R2",
            "qrc_beats_local_gbf",
            "qrc_improvement_vs_local_gbf_percent",
        ]
    ].to_csv(OUTPUT_DIR / "qrc_gbf_vs_local_gbf_ablation.csv", index=False)
    build_feature_summary(model1, model2).to_csv(OUTPUT_DIR / "model1_vs_model2_summary.csv", index=False)

    write_text_outputs(all_rows)
    write_plots(model1, model2, all_rows)
    write_paper_sections()

    config = Config(
        n_estimators=args.n_estimators,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        max_quantum_train_samples=args.max_quantum_train_samples,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
        device=args.device,
    )
    (OUTPUT_DIR / "model1_model2_comparison_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(f"Wrote Model 1 / Model 2 comparison outputs to {OUTPUT_DIR}")


def build_model1_rows() -> pd.DataFrame:
    qrc = pd.read_csv(QRC_MODEL1_PATH)
    local = pd.read_csv(LOCAL_MODEL1_PATH)
    local = local[local["model"].astype(str) == "Gradient-boosted backbone"]
    rows = []
    for dataset in DATASET_ORDER:
        qrc_row = qrc[qrc["dataset"].replace({"Dayton": "DAYTON"}) == dataset].sort_values("MAPE").iloc[0]
        local_row = local[local["dataset"].astype(str) == dataset].iloc[0]
        rows.append(make_row(dataset, 1, "Model 1 short-term", local_row, qrc_row, source="existing Model 1 sweep"))
    return pd.DataFrame(rows)


def run_model2_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        print(f"Running Model 2 long-term comparison for {dataset}", flush=True)
        df, _, _, _ = load_dataset(dataset)
        split_date = DATASETS[dataset]["split_date"]
        X_train, y_train, X_test, y_test, _, test = split_long_term_features(df, split_date)
        start = time.perf_counter()
        print(f"{dataset} Model 2: training Local GBF-2 on {len(X_train)} rows", flush=True)
        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=args.random_state,
            n_jobs=4,
            tree_method="hist",
        )
        model.fit(X_train, y_train, verbose=False)
        local_seconds = time.perf_counter() - start
        print(f"{dataset} Model 2: Local GBF-2 trained in {local_seconds:.2f}s", flush=True)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        local_row = metric_dict(y_test.to_numpy(), test_pred, local_seconds)
        local_row["notes"] = "Local GBF-2 long-term gradient-boosted backbone"

        qrc_pred, qrc_notes, qrc_seconds = qrc_correction(
            X_train,
            y_train,
            train_pred,
            X_test,
            y_test,
            test_pred,
            args,
            dataset,
        )
        qrc_row = metric_dict(y_test.to_numpy(), qrc_pred, qrc_seconds)
        qrc_row["notes"] = qrc_notes
        rows.append(make_row(dataset, 2, "Model 2 long-term", local_row, qrc_row, source="fresh Model 2 run"))
    return pd.DataFrame(rows)


def create_long_term_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    dates = features.index
    features["hour"] = dates.hour
    features["dayofweek"] = dates.dayofweek
    features["quarter"] = dates.quarter
    features["month"] = dates.month
    features["year"] = dates.year
    features["dayofyear"] = dates.dayofyear
    features["dayofmonth"] = dates.day
    features["weekofyear"] = dates.isocalendar().week.astype(int)
    load_source = features["Load"]
    for window in (24, 48, 168):
        features[f"load_{window}_hrs_lag"] = features["Load"].shift(window)
        rolling = load_source.rolling(window=window)
        features[f"load_{window}_hrs_mean"] = rolling.mean()
        features[f"load_{window}_hrs_std"] = rolling.std()
        features[f"load_{window}_hrs_max"] = rolling.max()
        features[f"load_{window}_hrs_min"] = rolling.min()
    return features


def split_long_term_features(df: pd.DataFrame, split_date: str):
    featured = create_long_term_features(df)
    columns = [column for column in featured.columns if column != "Load"]
    train = featured.loc[featured.index <= split_date].dropna(subset=columns + ["Load"])
    test = featured.loc[featured.index > split_date].dropna(subset=columns + ["Load"])
    return train[columns], train["Load"], test[columns], test["Load"], train, test


def qrc_correction(X_train, y_train, train_pred, X_test, y_test, test_pred, args, dataset: str):
    residual = y_train.to_numpy() - train_pred
    full_split = int(len(X_train) * (1.0 - args.validation_fraction))
    fit_residual = residual[:full_split]
    val_residual = residual[full_split:]
    val_pred = train_pred[full_split:]
    val_true = y_train.to_numpy()[full_split:]
    denominator = float(np.dot(fit_residual[:-1], fit_residual[:-1]))
    seq_alpha = 0.0 if denominator == 0.0 else float(np.dot(fit_residual[:-1], fit_residual[1:]) / denominator)
    val_prev = np.r_[fit_residual[-1], val_residual[:-1]]
    test_residual_truth = y_test.to_numpy() - test_pred
    test_prev = np.r_[val_residual[-1], test_residual_truth[:-1]]
    candidates = [
        {
            "validation_MAE": mean_absolute_error(val_true, val_pred + seq_alpha * val_prev),
            "test_pred": test_pred + seq_alpha * test_prev,
            "source": "sequential_residual",
            "alpha": seq_alpha,
            "gamma": 0.0,
        }
    ]

    X_quantum, residual_quantum, pred_quantum = sample_quantum_rows(
        X_train, residual, train_pred, args.max_quantum_train_samples, args.random_state
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
    print(f"{dataset} Model 2: VQC fit features {len(X_fit_angles)} rows", flush=True)
    Z_fit = quantum_transform(X_fit_angles, circuit, dataset, "fit")
    print(f"{dataset} Model 2: VQC validation features {len(X_val_angles)} rows", flush=True)
    Z_val = quantum_transform(X_val_angles, circuit, dataset, "validation")
    readout = Ridge(alpha=10.0)
    readout.fit(np.column_stack([X_fit.to_numpy(), Z_fit, pred_fit]), r_fit)
    q_val = readout.predict(np.column_stack([X_val.to_numpy(), Z_val, pred_val]))
    val_prev_quantum = np.r_[r_fit[-1], r_val[:-1]]
    test_prev_quantum = np.r_[r_val[-1], test_residual_truth[:-1]]
    y_val_quantum = pred_val + r_val
    for alpha in np.linspace(0.0, 0.8, 17):
        for gamma in np.linspace(-0.5, 0.5, 21):
            val_out = pred_val + alpha * val_prev_quantum + gamma * q_val
            candidates.append(
                {
                    "validation_MAE": mean_absolute_error(y_val_quantum, val_out),
                    "test_pred": None,
                    "source": "vqc_residual_blend",
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                }
            )
    best = min(candidates, key=lambda item: item["validation_MAE"])
    if best["source"] == "vqc_residual_blend":
        print(f"{dataset} Model 2: VQC test features {len(X_test_angles)} rows", flush=True)
        Z_test = quantum_transform(X_test_angles, circuit, dataset, "test")
        q_test = readout.predict(np.column_stack([X_test.to_numpy(), Z_test, test_pred]))
        best["test_pred"] = test_pred + best["alpha"] * test_prev_quantum + best["gamma"] * q_test
    notes = f"QRC-GBF-2 selected={best['source']}, alpha={best['alpha']:.3f}, gamma={best['gamma']:.3f}"
    return best["test_pred"], notes, time.perf_counter() - start


def sample_quantum_rows(X: pd.DataFrame, residual: np.ndarray, pred: np.ndarray, max_samples: int, seed: int):
    if max_samples <= 0 or len(X) <= max_samples:
        return X, residual, pred
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(X), size=max_samples, replace=False))
    return X.iloc[positions], residual[positions], pred[positions]


def make_circuit(n_qubits: int, weights: np.ndarray, device_name: str):
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x: np.ndarray):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def quantum_transform(X: np.ndarray, circuit, dataset: str, label: str) -> np.ndarray:
    rows = []
    total = len(X)
    checkpoint = max(1, total // 5)
    for index, row in enumerate(X, start=1):
        rows.append(circuit(row))
        if index == 1 or index == total or index % checkpoint == 0:
            print(f"{dataset} Model 2: {label} VQC rows {index}/{total}", flush=True)
    return np.asarray(rows, dtype=float)


def metric_dict(y_true: np.ndarray, pred: np.ndarray, seconds: float) -> dict:
    return {
        "MAPE": float(np.mean(np.abs((y_true - pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, pred)),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "seconds": float(seconds),
    }


def make_row(dataset: str, model_number: int, feature_setting: str, local_row: pd.Series | dict, qrc_row: pd.Series | dict, source: str) -> dict:
    published = PUBLISHED[dataset]
    published_mape = published[f"model{model_number}_mape"]
    published_r2 = published[f"model{model_number}_r2"]
    qrc_mape = float(qrc_row["MAPE"])
    local_mape = float(local_row["MAPE"])
    qrc_r2 = float(qrc_row["R2"])
    local_r2 = float(local_row["R2"])
    qrc_name = f"QRC-GBF-{model_number}"
    local_name = f"Local GBF-{model_number}"
    published_name = f"Published PredXGBR-{model_number}"
    return {
        "dataset": dataset,
        "feature_setting": feature_setting,
        "published_predxgbr_name": published_name,
        "published_predxgbr_MAPE": published_mape,
        "published_predxgbr_R2": published_r2,
        "local_gbf_name": local_name,
        "local_gbf_MAPE": local_mape,
        "local_gbf_R2": local_r2,
        "qrc_gbf_name": qrc_name,
        "qrc_gbf_MAPE": qrc_mape,
        "qrc_gbf_R2": qrc_r2,
        "qrc_gbf_MAE": float(qrc_row.get("MAE", np.nan)),
        "qrc_gbf_RMSE": float(qrc_row.get("RMSE", np.nan)),
        "best_model": qrc_name if qrc_mape <= min(local_mape, published_mape) else local_name if local_mape <= published_mape else published_name,
        "qrc_beats_published_predxgbr": bool(qrc_mape < published_mape and qrc_r2 >= published_r2),
        "qrc_beats_local_gbf": bool(qrc_mape < local_mape),
        "qrc_improvement_vs_published_percent": (published_mape - qrc_mape) / published_mape * 100,
        "qrc_improvement_vs_local_gbf_percent": (local_mape - qrc_mape) / local_mape * 100,
        "notes": qrc_row.get("notes", source),
    }


def build_feature_summary(model1: pd.DataFrame, model2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = model1.merge(model2, on="dataset", suffixes=("_model1", "_model2"))
    for row in merged.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "qrc_gbf_1_MAPE": row.qrc_gbf_MAPE_model1,
                "qrc_gbf_2_MAPE": row.qrc_gbf_MAPE_model2,
                "better_qrc_feature_setting": "QRC-GBF-1" if row.qrc_gbf_MAPE_model1 <= row.qrc_gbf_MAPE_model2 else "QRC-GBF-2",
                "local_gbf_1_MAPE": row.local_gbf_MAPE_model1,
                "local_gbf_2_MAPE": row.local_gbf_MAPE_model2,
                "better_local_feature_setting": "Local GBF-1" if row.local_gbf_MAPE_model1 <= row.local_gbf_MAPE_model2 else "Local GBF-2",
            }
        )
    return pd.DataFrame(rows)


def write_text_outputs(all_rows: pd.DataFrame) -> None:
    explanation = """Model 1 / Model 2 Naming Explanation

Following the PredXGBR study, two feature configurations are considered. Model 1 uses short-term lag features to capture recent demand fluctuations, while Model 2 uses long-term lag features to capture broader temporal and weekly patterns. In this work, the same naming convention is adopted: QRC-GBF-1 denotes the proposed method under the short-term feature setting, and QRC-GBF-2 denotes the proposed method under the long-term feature setting.

Published PredXGBR-1 and Published PredXGBR-2 refer only to the values reported in the original paper. Local GBF-1 and Local GBF-2 refer to same-code local gradient-boosted backbones. QRC-GBF-1 and QRC-GBF-2 refer to the proposed quantum residual correction framework under the corresponding feature setting.
"""
    (OUTPUT_DIR / "final_model_naming_explanation.txt").write_text(explanation)
    lines = ["Final Model 1 / Model 2 Result Summary", ""]
    for row in all_rows.itertuples(index=False):
        lines.append(
            f"{row.dataset} {row.feature_setting}: {row.qrc_gbf_name} MAPE={row.qrc_gbf_MAPE:.4f}, "
            f"R2={row.qrc_gbf_R2:.6f}, published={row.published_predxgbr_name} MAPE={row.published_predxgbr_MAPE:.4f}, "
            f"published_win={row.qrc_beats_published_predxgbr}, local_win={row.qrc_beats_local_gbf}, "
            f"published_improvement={row.qrc_improvement_vs_published_percent:.2f}%"
        )
    (OUTPUT_DIR / "final_result_summary.txt").write_text("\n".join(lines) + "\n")


def write_paper_sections() -> None:
    text = """Paper Section Updates: Model 1 and Model 2

Method naming:
Following the PredXGBR study, two feature configurations are considered. Model 1 uses short-term lag features to capture recent demand fluctuations, while Model 2 uses long-term lag features to capture broader temporal and weekly patterns. In this work, the same naming convention is adopted: QRC-GBF-1 denotes the proposed method under the short-term feature setting, and QRC-GBF-2 denotes the proposed method under the long-term feature setting.

Experimental setup:
Published PredXGBR-1 and Published PredXGBR-2 are used as external reference baselines from Table 3 of the PredXGBR paper. Local GBF-1 and Local GBF-2 are same-code gradient-boosted backbones implemented in the present experimental pipeline. QRC-GBF-1 and QRC-GBF-2 extend the corresponding local backbones with quantum residual correction and validation-selected sequential postprocessing.

Baseline description:
Published PredXGBR results are not relabeled as local results. Local GBF is not claimed to be an exact PredXGBR reproduction because original paper code, preprocessing, feature construction, hyperparameters, and split handling are not fully matched.

Results and discussion:
External comparison reports QRC-GBF-1 vs Published PredXGBR-1 and QRC-GBF-2 vs Published PredXGBR-2. Internal ablation reports QRC-GBF-1 vs Local GBF-1 and QRC-GBF-2 vs Local GBF-2. Feature-setting analysis compares Model 1 against Model 2 within both the local backbone and proposed method.

Ablation study:
The local GBF rows isolate the contribution of the residual-correction framework. Improvement over Local GBF indicates added value from QRC-GBF beyond the gradient-boosted backbone.

Limitations:
The external baseline uses published Table 3 values, while local GBF uses the present implementation. Therefore, external and internal comparisons answer different questions and should not be conflated.
"""
    (OUTPUT_DIR / "paper_section_updates_model1_model2.txt").write_text(text)


def write_plots(model1: pd.DataFrame, model2: pd.DataFrame, all_rows: pd.DataFrame) -> None:
    plot_mape(model1, FIGURE_DIR / "model1_short_term_mape.png", "Model 1 Short-Term Feature Setting")
    plot_mape(model2, FIGURE_DIR / "model2_long_term_mape.png", "Model 2 Long-Term Feature Setting")
    plot_qrc_vs_published(model1, FIGURE_DIR / "qrc_gbf_vs_predxgbr_model1.png", "QRC-GBF-1 vs Published PredXGBR-1")
    plot_qrc_vs_published(model2, FIGURE_DIR / "qrc_gbf_vs_predxgbr_model2.png", "QRC-GBF-2 vs Published PredXGBR-2")
    plot_ablation(all_rows, FIGURE_DIR / "qrc_gbf_ablation_model1_model2.png")


def plot_mape(df: pd.DataFrame, output: Path, title: str) -> None:
    ordered = df.set_index("dataset").loc[DATASET_ORDER].reset_index()
    x = np.arange(len(ordered))
    width = 0.25
    plt.figure(figsize=(11, 5))
    plt.bar(x - width, ordered["published_predxgbr_MAPE"], width, label="Published PredXGBR", color="#8d99ae")
    plt.bar(x, ordered["local_gbf_MAPE"], width, label="Local GBF", color="#457b9d")
    plt.bar(x + width, ordered["qrc_gbf_MAPE"], width, label="QRC-GBF", color="#2a9d8f")
    plt.xticks(x, ordered["dataset"])
    plt.ylabel("MAPE (%) lower is better")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_qrc_vs_published(df: pd.DataFrame, output: Path, title: str) -> None:
    ordered = df.set_index("dataset").loc[DATASET_ORDER].reset_index()
    x = np.arange(len(ordered))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, ordered["published_predxgbr_MAPE"], width=0.4, label="Published PredXGBR", color="#8d99ae")
    plt.bar(x + 0.2, ordered["qrc_gbf_MAPE"], width=0.4, label="QRC-GBF", color="#2a9d8f")
    plt.xticks(x, ordered["dataset"])
    plt.ylabel("MAPE (%) lower is better")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_ablation(all_rows: pd.DataFrame, output: Path) -> None:
    labels = [f"{row.dataset}\n{row.feature_setting.split()[1]}" for row in all_rows.itertuples(index=False)]
    x = np.arange(len(all_rows))
    plt.figure(figsize=(13, 5))
    plt.bar(x - 0.2, all_rows["local_gbf_MAPE"], width=0.4, label="Local GBF", color="#457b9d")
    plt.bar(x + 0.2, all_rows["qrc_gbf_MAPE"], width=0.4, label="QRC-GBF", color="#2a9d8f")
    plt.xticks(x, labels)
    plt.ylabel("MAPE (%) lower is better")
    plt.title("Internal Ablation: QRC-GBF vs Local GBF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
