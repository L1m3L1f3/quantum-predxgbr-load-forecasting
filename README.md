# PredXGBR Quantum Load Forecasting

**GitHub repository name:** `quantum-predxgbr-load-forecasting`

**GitHub description:** Hybrid quantum-classical XGBoost experiments for short-term electricity load forecasting in power systems.

**GitHub topics/tags:** `quantum-machine-learning`, `hybrid-quantum-classical`, `xgboost`, `load-forecasting`, `time-series-forecasting`, `power-systems`, `energy-forecasting`, `pennylane`, `qrc-gbf`, `python`

This repository contains the original PredXGBR paper code plus a reproducible terminal runner for using XGBoost as the classical AI baseline in:

> Classical and Hybrid Quantum Machine Learning for Short-Term Electricity Load Forecasting in Power Systems.

The original paper scripts are kept under `PredXGBR/` and were not deleted. New runner files at the repository root make the project easier to execute from a fresh Python environment.

## Paper and Code Source

- Original code repository: <https://github.com/rifatzabin/PredXGBR>
- Related paper: Rifat Zabin, Khandaker Foysal Haque, and Ahmed Abdelgawad, "PredXGBR: A Machine Learning Framework for Short-Term Electrical Load Prediction", Electronics 2024, 13, 4521. DOI: <https://doi.org/10.3390/electronics13224521>

## Repository Structure

```text
.
|-- PredXGBR/
|   |-- AEP/              # Original AEP scripts
|   |-- Code/             # Original generic/PJM scripts
|   |-- DAYTON/           # Original DAYTON scripts
|   |-- PJM/              # Original PJM scripts
|   |-- PJME/             # Original PJME scripts
|   `-- PJMW/PJMW/        # Original PJMW scripts
|-- baseline_pipeline.py  # Reproducible preprocessing/training/evaluation pipeline
|-- run_preprocessing.py  # Terminal preprocessing entry point
|-- run_training.py       # Terminal training entry point
|-- run_evaluation.py     # Terminal evaluation entry point
|-- run_all.py            # Runs preprocessing, training, and evaluation
|-- run_quantum.py        # Runs the GPU-backed hybrid quantum feature baseline
|-- Forecasting_Data/     # Preferred local dataset CSVs downloaded outside Git
|-- data/                 # Fallback local dataset CSVs
|-- processed/            # Generated train/test feature tables
|-- models/               # Generated trained XGBoost model
|-- results/              # Generated metrics, predictions, and plots
`-- requirements.txt      # Python dependencies
```

## Original Script Notes

Each original dataset folder has the same pattern:

- `packages.py`: shared imports.
- `datagen.py`: reads a CSV, splits train/test by date, and creates time features plus lag/rolling load features.
- `train_xgboost_2.py`: trains `xgb.XGBRegressor(n_estimators=1000)` and saves a pickle model.
- `results.py`: loads the model, predicts the test set, and prints metrics.
- `datagen_plot.py` and `results_plot.py`: plotting-oriented versions of preprocessing and evaluation.

The original scripts hard-code paths like `../../../PJM_Load/PJME_hourly.csv`, so they do not run directly from this cloned repository unless the external `PJM_Load` folder is manually recreated. The root-level runner searches the locally downloaded `Forecasting_Data/` folder first and falls back to `data/`.

## Dataset

The default baseline uses `PJME_hourly.csv` from the public Kaggle dataset "Hourly Energy Consumption" by Rob Mulla:

- Dataset page: <https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption>
- License: CC0/Public Domain
- Units: megawatts (MW)
- Datetime column: `Datetime`
- Prediction target for PJME: `PJME_MW`
- Internal normalized target name in the runner: `Load`

The checked runner also supports the original project's dataset names when their files are placed in the preferred local `Forecasting_Data/` folder or the fallback `data/` folder:

| Dataset | Preferred local file | Fallback local file | Target column |
| --- | --- | --- | --- |
| PJME | `Forecasting_Data/PJME_hourly.csv` | `data/PJME_hourly.csv` | `PJME_MW` or `Load` |
| PJM | `Forecasting_Data/PJM_Load_hourly.csv` | `data/PJM_Load_hourly.csv` | `PJM_Load_MW` or `Load` |
| AEP | `Forecasting_Data/AEP_hourly.csv` | `data/AEP_hourly.csv` | `AEP_MW` or `Load` |
| DAYTON | `Forecasting_Data/DAYTON_hourly.csv` | `data/DAYTON_hourly.csv` | `DAYTON_MW` or `Load` |
| PJMW | `Forecasting_Data/PJMW_hourly.csv` | `data/PJMW_hourly.csv` | `PJMW_MW` or `Load` |

If a file is missing, place the required CSV in `Forecasting_Data/` or `data/`, or pass `--data-file /path/to/file.csv`.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the Baseline

After cloning/accessing this repository from GitHub and downloading the dataset files into `Forecasting_Data/`, run the complete default PJME experiment:

```bash
.venv/bin/python run_all.py
```

Equivalent staged commands:

```bash
.venv/bin/python run_preprocessing.py
.venv/bin/python run_training.py
.venv/bin/python run_evaluation.py
```

Run another supported dataset after placing its CSV in `Forecasting_Data/` or `data/`:

```bash
.venv/bin/python run_all.py --dataset AEP
```


## Export Results to Excel and Graphs

After a model run completes, create a downloadable Excel workbook plus extra visualization graphs from the generated `results/` files:

```bash
.venv/bin/python scripts/export_model_results.py
```

The exporter writes `reports/predxgbr_local_dataset_results.xlsx` with `metrics`, `run_config`, `predictions`, and `training_history` sheets. It also writes residual-distribution and daily-absolute-error PNG graphs in `reports/`.

## Run the Quantum Baseline

The repository now includes a first runnable hybrid quantum baseline using PennyLane `lightning.gpu`. It reuses the classical feature table, reduces features to the number of qubits with PCA, encodes them into a simulated quantum circuit, and trains a classical ridge readout on the resulting quantum expectation features.

Run a GPU quantum experiment:

```bash
.venv/bin/python run_quantum.py --device lightning.gpu --n-qubits 4 --n-layers 2 --max-train-samples 1000 --max-test-samples 500
```

Outputs are saved in `results/quantum/`:

- `quantum_metrics.csv`
- `quantum_predictions.csv`
- `quantum_actual_vs_predicted.png`
- `quantum_run_config.json`

The quantum model artifact is saved in `models/quantum/`.

## Methodology

- Model: XGBoost regression (`XGBRegressor`)
- Default dataset: PJME hourly load
- Forecasting target: current hourly load in MW
- Forecasting horizon: 1 hour ahead in the root runner
- Train/test split: train timestamps `<= 2015-01-02`, test timestamps `> 2015-01-02` for PJME
- Input features:
  - Calendar features: hour, day of week, quarter, month, year, day of year, day of month, ISO week of year
  - Historical load features: 6, 12, and 24 hour lags
  - Historical rolling statistics: 6, 12, and 24 hour mean, standard deviation, max, and min
- Evaluation metrics: MAE, RMSE, MAPE, R2, training time

Important reproducibility note: the original paper scripts compute rolling features directly from `Load`, which includes the current target value in rolling mean/max/min statistics. The root runner defaults to `--feature-mode causal`, where rolling features are shifted by one hour before calculation to avoid target leakage. To reproduce the original behavior for comparison, run with `--feature-mode original` and document that setting.

## Outputs

After `run_all.py`, outputs are saved in `results/`:

- `metrics.csv`: MAE, RMSE, MAPE, R2, and training time.
- `predictions.csv`: timestamp, actual load, predicted load.
- `training_history.csv`: train/test RMSE per boosting iteration.
- `actual_vs_predicted.png`: full test-period actual vs predicted plot.
- `loss_curve.png`: XGBoost train/test RMSE curve.
- `run_config.json`: dataset, target, split, row counts, and feature mode.

Generated model and feature artifacts:

- `models/xgboost_<dataset>.pkl`
- `processed/<dataset>_train_features.csv`
- `processed/<dataset>_test_features.csv`

## Verified Result

On the downloaded `PJME_hourly.csv` with causal features:

| Dataset | MAE | RMSE | MAPE | R2 | Training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| PJME | 511.742 | 707.359 | 1.623% | 0.987981 | 54.86 s |

Verified GPU quantum smoke comparison on 1000 train samples and 500 test samples:

| Dataset | Model | MAE | RMSE | MAPE | R2 | Training time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PJME | hybrid quantum feature ridge | 5410.942 | 6624.630 | 18.385% | 0.006010 | 4.04 s |

## Future Hybrid Quantum Extension

Do not replace this baseline when adding quantum models. Reuse the same dataset split, target, feature engineering policy, and metrics so the comparison is fair. Add quantum or hybrid quantum code in separate files or folders, then write its predictions and metrics to a separate results file such as `results/quantum_metrics.csv`.

See `quantum_extension_plan.md` for the proposed comparison plan.
