# QRC-GBF Full Paper Ready Package

This folder contains a complete paper draft for:

**QRC-GBF: Quantum Residual Correction Gradient-Boosted Forecasting for Short-Term Electrical Load Prediction**

Main file:

- `main.tex`

Figures are stored locally in:

- `figures/`

Bibliography:

- `references.bib`

Main paper claim:

QRC-GBF improves over the published PredXGBR reference baselines across all five datasets. The internal same-code ablation is mixed, so the paper does not claim standalone quantum advantage. The safe claim is that QRC-GBF is a hybrid residual-correction framework with a quantum-enhanced residual candidate and validation-selected correction.

Important result summary:

- QRC-GBF-1 beats Published PredXGBR-1 on all five datasets.
- QRC-GBF-2 beats Published PredXGBR-2 on all five datasets.
- Robustness sweep: 20/20 wins against published reference baselines.
- Internal local backbone ablation: 11/20 wins.
- Best overall method: QRC-GBF-1.

Use this folder for Overleaf upload or local LaTeX editing.
