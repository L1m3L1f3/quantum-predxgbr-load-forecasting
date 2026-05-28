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
import pennylane as qml
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_pipeline import DATASETS, RESULTS_DIR, load_dataset


OUTPUT_DIR = RESULTS_DIR / "gnn_vqc_extension"
DEFAULT_NODES = ["PJME", "PJMW", "AEP", "DAYTON"]


@dataclass
class GNNVQCConfig:
    nodes: list[str]
    split_date: str
    max_rows: int
    max_vqc_train_samples: int
    max_vqc_eval_samples: int
    epochs: int
    batch_size: int
    hidden_size: int
    n_qubits: int
    n_layers: int
    quantum_device: str
    torch_device: str
    random_state: int


class GraphConvForecast(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.input = nn.Linear(n_features, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def graph_mix(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjf->bif", self.adjacency, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input(self.graph_mix(x)))
        h = torch.relu(self.hidden(self.graph_mix(h)))
        return self.output(h).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-capable GNN-VQC extension for overlapping multi-region load forecasting.")
    parser.add_argument("--nodes", nargs="+", default=DEFAULT_NODES, choices=sorted(DATASETS))
    parser.add_argument("--split-date", default="2015-01-02")
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--max-vqc-train-samples", type=int, default=1000)
    parser.add_argument("--max-vqc-eval-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--quantum-device", default="lightning.qubit")
    parser.add_argument("--torch-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_aligned_loads(nodes: list[str]) -> pd.DataFrame:
    series = []
    for node in nodes:
        df, _, _, _ = load_dataset(node)
        df = df.sort_index()
        if df.index.has_duplicates:
            df = df.groupby(level=0).mean()
        series.append(df.rename(columns={"Load": node}))
    aligned = pd.concat(series, axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError(
            "No overlapping timestamps for selected nodes. Use a valid overlapping subset, "
            "for example PJME PJMW AEP DAYTON."
        )
    return aligned.sort_index()


def make_features(loads: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    frames = []
    for node in loads.columns:
        features = pd.DataFrame(index=loads.index)
        features["load"] = loads[node]
        features["hour"] = loads.index.hour
        features["dayofweek"] = loads.index.dayofweek
        features["month"] = loads.index.month
        for lag in (1, 6, 12, 24):
            features[f"lag_{lag}"] = loads[node].shift(lag)
        for window in (6, 12, 24):
            source = loads[node].shift(1)
            features[f"mean_{window}"] = source.rolling(window).mean()
            features[f"std_{window}"] = source.rolling(window).std()
        frames.append(features)

    valid_index = frames[0].dropna().index
    for frame in frames[1:]:
        valid_index = valid_index.intersection(frame.dropna().index)

    feature_names = [column for column in frames[0].columns if column != "load"]
    X = np.stack([frame.loc[valid_index, feature_names].to_numpy(dtype=np.float32) for frame in frames], axis=1)
    y = loads.loc[valid_index].to_numpy(dtype=np.float32)
    return X, y, valid_index, feature_names


def normalize_adjacency(train_loads: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(train_loads.T)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    np.fill_diagonal(corr, 1.0)
    degree = corr.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-8)))
    return d_inv_sqrt @ corr @ d_inv_sqrt


def metric_row(model: str, y_true: np.ndarray, pred: np.ndarray, seconds: float, notes: str) -> dict:
    y_flat = y_true.reshape(-1)
    p_flat = pred.reshape(-1)
    return {
        "model": model,
        "MAPE": float(np.mean(np.abs((y_flat - p_flat) / y_flat)) * 100),
        "R2": float(r2_score(y_flat, p_flat)),
        "MAE": float(mean_absolute_error(y_flat, p_flat)),
        "RMSE": float(np.sqrt(mean_squared_error(y_flat, p_flat))),
        "seconds": float(seconds),
        "notes": notes,
    }


def make_circuit(n_qubits: int, n_layers: int, device_name: str, seed: int):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(low=-np.pi, high=np.pi, size=(n_layers, n_qubits, 3))
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit, weights


def quantum_transform(X: np.ndarray, circuit, label: str) -> np.ndarray:
    rows = []
    total = len(X)
    checkpoint = max(1, total // 5)
    for index, row in enumerate(X, start=1):
        rows.append(circuit(row))
        if index == 1 or index == total or index % checkpoint == 0:
            print(f"{label}: VQC rows {index}/{total}", flush=True)
    return np.asarray(rows, dtype=float)


def run(args: argparse.Namespace) -> pd.DataFrame:
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loads = load_aligned_loads(args.nodes)
    if args.max_rows > 0 and len(loads) > args.max_rows:
        loads = loads.iloc[-args.max_rows :]
    X, y, index, feature_names = make_features(loads)
    train_mask = index <= args.split_date
    test_mask = index > args.split_date
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train/test split produced empty data. Adjust --split-date or --max-rows.")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    n_train, n_nodes, n_features = X_train.shape
    X_train_scaled = x_scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, n_nodes, n_features)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, n_features)).reshape(len(X_test), n_nodes, n_features)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)

    adjacency_np = normalize_adjacency(y_train)
    torch_device = torch.device(args.torch_device)
    model = GraphConvForecast(
        n_features=n_features,
        hidden_size=args.hidden_size,
        adjacency=torch.tensor(adjacency_np, dtype=torch.float32, device=torch_device),
    ).to(torch_device)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    print(f"Training GNN on {torch_device} with nodes={args.nodes}, train={len(X_train)}, test={len(X_test)}", flush=True)
    start = time.perf_counter()
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for xb, yb in loader:
            xb = xb.to(torch_device)
            yb = yb.to(torch_device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 4) == 0:
            print(f"epoch={epoch} gnn_loss={np.mean(losses):.6f}", flush=True)

    gnn_pred_train = predict_gnn(model, X_train_scaled, y_scaler, args.batch_size, torch_device)
    gnn_pred_test = predict_gnn(model, X_test_scaled, y_scaler, args.batch_size, torch_device)
    gnn_seconds = time.perf_counter() - start

    train_residual = y_train - gnn_pred_train
    train_features_flat = np.column_stack([X_train_scaled.reshape(-1, n_features), gnn_pred_train.reshape(-1)])
    test_features_flat = np.column_stack([X_test_scaled.reshape(-1, n_features), gnn_pred_test.reshape(-1)])
    train_residual_flat = train_residual.reshape(-1)

    rng = np.random.default_rng(args.random_state)
    train_positions = np.arange(len(train_features_flat))
    if args.max_vqc_train_samples > 0 and len(train_positions) > args.max_vqc_train_samples:
        train_positions = np.sort(rng.choice(train_positions, size=args.max_vqc_train_samples, replace=False))
    eval_positions = np.arange(len(test_features_flat))
    if args.max_vqc_eval_samples > 0 and len(eval_positions) > args.max_vqc_eval_samples:
        eval_positions = np.sort(rng.choice(eval_positions, size=args.max_vqc_eval_samples, replace=False))

    preprocessing = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=args.n_qubits, random_state=args.random_state)),
            ("angle_scale", MinMaxScaler(feature_range=(-np.pi, np.pi))),
        ]
    )
    X_vqc_train = preprocessing.fit_transform(train_features_flat[train_positions])
    X_vqc_eval = preprocessing.transform(test_features_flat[eval_positions])
    circuit, weights = make_circuit(args.n_qubits, args.n_layers, args.quantum_device, args.random_state)
    vqc_start = time.perf_counter()
    Z_train = quantum_transform(X_vqc_train, circuit, "train")
    Z_eval = quantum_transform(X_vqc_eval, circuit, "eval")
    readout = Ridge(alpha=10.0)
    readout.fit(Z_train, train_residual_flat[train_positions])
    residual_eval = readout.predict(Z_eval)

    y_test_flat = y_test.reshape(-1)
    gnn_test_flat = gnn_pred_test.reshape(-1)
    gnn_vqc_eval_pred = gnn_test_flat[eval_positions] + residual_eval
    vqc_seconds = time.perf_counter() - vqc_start

    rows = [
        metric_row("GNN", y_test, gnn_pred_test, gnn_seconds, "manual PyTorch graph convolution over overlapping regions"),
        metric_row(
            "GNN-VQC residual",
            y_test_flat[eval_positions],
            gnn_vqc_eval_pred,
            gnn_seconds + vqc_seconds,
            "GNN backbone plus fixed VQC residual features and Ridge residual readout on sampled eval rows",
        ),
    ]
    results = pd.DataFrame(rows)
    results.insert(0, "nodes", ",".join(args.nodes))
    results.to_csv(OUTPUT_DIR / "gnn_vqc_extension_metrics.csv", index=False)
    pd.DataFrame(adjacency_np, index=args.nodes, columns=args.nodes).to_csv(OUTPUT_DIR / "gnn_vqc_adjacency.csv")
    (OUTPUT_DIR / "gnn_vqc_extension_config.json").write_text(json.dumps(asdict(config_from_args(args)), indent=2) + "\n")
    write_plots(results, y_test_flat[eval_positions], gnn_test_flat[eval_positions], gnn_vqc_eval_pred)
    print(results.to_string(index=False))
    return results


def predict_gnn(model, X: np.ndarray, y_scaler: StandardScaler, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).cpu().numpy())
    pred_scaled = np.concatenate(preds, axis=0)
    return y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(pred_scaled.shape)


def write_plots(results: pd.DataFrame, actual: np.ndarray, gnn_pred: np.ndarray, gnn_vqc_pred: np.ndarray) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(results["model"], results["MAPE"], color=["#457b9d", "#2a9d8f"])
    plt.ylabel("MAPE (%) lower is better")
    plt.title("GNN-VQC Extension")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gnn_vqc_extension_mape.png", dpi=160)
    plt.close()

    view = min(500, len(actual))
    plt.figure(figsize=(12, 4))
    plt.plot(actual[:view], label="Actual", linewidth=1)
    plt.plot(gnn_pred[:view], label="GNN", linewidth=1)
    plt.plot(gnn_vqc_pred[:view], label="GNN-VQC residual", linewidth=1)
    plt.legend()
    plt.title("GNN-VQC Forecast Sample")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gnn_vqc_extension_forecast.png", dpi=160)
    plt.close()


def config_from_args(args: argparse.Namespace) -> GNNVQCConfig:
    return GNNVQCConfig(
        nodes=list(args.nodes),
        split_date=args.split_date,
        max_rows=args.max_rows,
        max_vqc_train_samples=args.max_vqc_train_samples,
        max_vqc_eval_samples=args.max_vqc_eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        quantum_device=args.quantum_device,
        torch_device=args.torch_device,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    run(parse_args())
