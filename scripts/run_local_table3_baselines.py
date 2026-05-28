from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.compose import TransformedTargetRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_pipeline import DATASETS, RESULTS_DIR, load_dataset, split_features


OUTPUT_DIR = RESULTS_DIR / "local_table3_baselines"
QRC_BEST_PATH = RESULTS_DIR / "qrc_gbf" / "sweep" / "qrc_gbf_sweep_best_by_dataset.csv"
DATASET_ORDER = ["PJM", "PJME", "PJMW", "AEP", "DAYTON"]
MODEL_ORDER = ["SVM", "RNN", "LSTM", "TCN", "Transformer", "Gradient-boosted backbone", "QRC-GBF"]


@dataclass
class LocalBaselineConfig:
    datasets: list[str]
    feature_mode: str
    epochs: int
    batch_size: int
    max_train_rows: int
    random_state: int


class RNNRegressor(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class TCNRegressor(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.net(x).squeeze(-1)
        return self.head(out).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 4):
        super().__init__()
        self.project = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        out = self.encoder(self.project(x))
        return self.head(out.mean(dim=1)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Table 3-style baselines against QRC-GBF.")
    parser.add_argument("--datasets", nargs="+", default=DATASET_ORDER, choices=DATASET_ORDER)
    parser.add_argument("--feature-mode", choices=["causal", "original"], default="original")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-rows", type=int, default=30000)
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
        "seconds": seconds,
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    qrc = load_qrc_best()
    all_rows = []
    for dataset in args.datasets:
        print(f"Running local baselines for {dataset}", flush=True)
        all_rows.extend(run_dataset(dataset, args, qrc))

    results = pd.DataFrame(all_rows)
    results["model"] = pd.Categorical(results["model"], MODEL_ORDER, ordered=True)
    results["dataset"] = pd.Categorical(results["dataset"], DATASET_ORDER, ordered=True)
    results = results.sort_values(["dataset", "model"])
    results.to_csv(OUTPUT_DIR / "local_table3_baselines_results.csv", index=False)
    average = results.groupby("model", observed=True).agg(average_MAPE=("MAPE", "mean"), average_R2=("R2", "mean")).reset_index()
    average.to_csv(OUTPUT_DIR / "local_table3_baselines_average.csv", index=False)
    write_summary(results, average, args)
    write_plots(results, average)
    config = LocalBaselineConfig(
        datasets=list(args.datasets),
        feature_mode=args.feature_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_rows=args.max_train_rows,
        random_state=args.random_state,
    )
    (OUTPUT_DIR / "local_table3_baselines_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(results.to_string(index=False))


def load_qrc_best() -> pd.DataFrame:
    if not QRC_BEST_PATH.exists():
        raise FileNotFoundError(f"Missing QRC-GBF best results: {QRC_BEST_PATH}. Run scripts/run_qrc_gbf_sweep.py first.")
    qrc = pd.read_csv(QRC_BEST_PATH)
    qrc["dataset"] = qrc["dataset"].replace({"Dayton": "DAYTON"})
    return qrc


def run_dataset(dataset: str, args: argparse.Namespace, qrc: pd.DataFrame) -> list[dict]:
    split_date = DATASETS[dataset]["split_date"]
    df, _, _, _ = load_dataset(dataset)
    X_train, y_train, X_test, y_test, _, _ = split_features(df, split_date, args.feature_mode)
    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    rows = []
    rows.append(run_svm(dataset, X_train, y_train_np, X_test, y_test_np, args))
    rows.append(run_xgboost_backbone(dataset, X_train, y_train_np, X_test, y_test_np, args))

    X_fit, y_fit = sample_rows(X_train, y_train_np, args.max_train_rows, args.random_state)
    for name, factory in [
        ("RNN", RNNRegressor),
        ("LSTM", LSTMRegressor),
        ("TCN", TCNRegressor),
        ("Transformer", TransformerRegressor),
    ]:
        rows.append(run_torch_model(name, factory(), dataset, X_fit, y_fit, X_test, y_test_np, args))

    qrc_row = qrc[qrc["dataset"] == dataset].sort_values("MAPE").iloc[0]
    rows.append(
        {
            "dataset": dataset,
            "model": "QRC-GBF",
            "MAPE": float(qrc_row["MAPE"]),
            "R2": float(qrc_row["R2"]),
            "MAE": float(qrc_row["MAE"]),
            "RMSE": float(qrc_row["RMSE"]),
            "seconds": float(qrc_row.get("seconds", np.nan)),
            "notes": "best QRC-GBF setting from VQC sweep",
        }
    )
    return rows


def sample_rows(X: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(X), size=max_rows, replace=False))
    return X.iloc[positions], y[positions]


def run_svm(dataset: str, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, args: argparse.Namespace) -> dict:
    X_fit, y_fit = sample_rows(X_train, y_train, args.max_train_rows, args.random_state)
    start = time.perf_counter()
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                TransformedTargetRegressor(
                    regressor=LinearSVR(C=1.0, epsilon=0.0, max_iter=5000, random_state=args.random_state),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )
    model.fit(X_fit, y_fit)
    pred = model.predict(X_test)
    return metric_row(dataset, "SVM", y_test, pred, time.perf_counter() - start, "local LinearSVR on same engineered features")


def run_xgboost_backbone(dataset: str, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    model = xgb.XGBRegressor(
        n_estimators=1000,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=args.random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    return metric_row(dataset, "Gradient-boosted backbone", y_test, pred, time.perf_counter() - start, "local gradient-boosted backbone")


def run_torch_model(
    name: str,
    model: nn.Module,
    dataset: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    start = time.perf_counter()
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_fit = x_scaler.fit_transform(X_train).astype(np.float32)
    X_eval = x_scaler.transform(X_test).astype(np.float32)
    y_fit = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)

    train_tensor = torch.tensor(X_fit).unsqueeze(-1)
    target_tensor = torch.tensor(y_fit)
    loader = DataLoader(TensorDataset(train_tensor, target_tensor), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for start_idx in range(0, len(X_eval), args.batch_size):
            batch = torch.tensor(X_eval[start_idx : start_idx + args.batch_size]).unsqueeze(-1)
            preds.append(model(batch).cpu().numpy())
    pred_scaled = np.concatenate(preds)
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    return metric_row(
        dataset,
        name,
        y_test,
        pred,
        time.perf_counter() - start,
        f"local PyTorch {name}; train_rows={len(X_train)}, epochs={args.epochs}",
    )


def write_summary(results: pd.DataFrame, average: pd.DataFrame, args: argparse.Namespace) -> None:
    best = average.sort_values("average_MAPE").iloc[0]
    lines = [
        "Local Table 3-Style Baseline Comparison",
        "",
        f"Feature mode: {args.feature_mode}",
        f"Deep model epochs: {args.epochs}",
        f"Max train rows for SVM/deep models: {args.max_train_rows}",
        "",
        "Important note: local RNN/LSTM/TCN/Transformer use compact PyTorch implementations on the same engineered feature rows.",
        "They are same-code local baselines, not guaranteed exact reproductions of the PredXGBR paper architectures.",
        "",
        f"Best average MAPE model: {best.model} ({best.average_MAPE:.4f})",
        "",
        average.sort_values("average_MAPE").to_string(index=False),
    ]
    (OUTPUT_DIR / "local_table3_baselines_summary.txt").write_text("\n".join(lines) + "\n")


def write_plots(results: pd.DataFrame, average: pd.DataFrame) -> None:
    present_datasets = [dataset for dataset in DATASET_ORDER if dataset in set(results["dataset"].astype(str))]
    pivot = results.pivot(index="dataset", columns="model", values="MAPE").loc[present_datasets, MODEL_ORDER]
    x = np.arange(len(present_datasets))
    width = 0.11
    colors = ["#8d99ae", "#b8b8d1", "#b0c4b1", "#f4a261", "#90be6d", "#457b9d", "#2a9d8f"]
    plt.figure(figsize=(14, 6))
    for i, model in enumerate(MODEL_ORDER):
        plt.bar(x + (i - 3) * width, pivot[model], width=width, label=model, color=colors[i])
    plt.xticks(x, present_datasets)
    plt.ylabel("MAPE (%) lower is better")
    plt.title("Local Same-Code Baselines vs QRC-GBF")
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "local_table3_baselines_mape_by_dataset.png", dpi=180)
    plt.close()

    ordered = average.sort_values("average_MAPE")
    colors = ["#2a9d8f" if str(model) == "QRC-GBF" else "#8d99ae" for model in ordered["model"].astype(str)]
    plt.figure(figsize=(10, 5))
    plt.bar(ordered["model"].astype(str), ordered["average_MAPE"], color=colors)
    plt.ylabel("Average MAPE (%) lower is better")
    plt.title("Local Same-Code Average MAPE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "local_table3_baselines_average_mape.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
