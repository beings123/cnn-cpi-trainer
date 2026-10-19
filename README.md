# Common Prosperity Index (CPI) Prediction System · 共同富裕指数预测系统

A pure-NumPy **2D-CNN + LSTM** model that predicts next-year Common Prosperity Index (CPI) for 31 Chinese provinces.

一个基于纯 NumPy 的 **2D-CNN + LSTM** 模型，用于预测中国 31 个省份次年的共同富裕指数（CPI）。

> Full technical report: [report.html](report.html) · 完整技术报告见 [report.html](report.html)

---

## Overview · 概述

| Item · 项目 | Value · 值 |
|---|---|
| Task · 任务 | Regression forecast of a composite prosperity index · 综合指数的回归预测 |
| Scope · 范围 | 31 provinces × 8 indicators · 31 省 × 8 指标 |
| Data · 数据 | 2013–2023 (341 samples), National Bureau of Statistics · 国家统计局 |
| Model · 模型 | 2D-CNN + LSTM (7,281 parameters, pure NumPy) |
| Best result · 最佳结果 | Val MSE 0.003717 @ epoch 71 |

## Pipeline · 流程

```
fetch_real_data.py ──► common_prosperity_data.csv ──► train_cnn.py ──► best_cnn_lstm_model.h5 ──► predict.py
   (real NBS data)      (8 indicators + CPI)          (2D-CNN + LSTM)         (trained weights)      (next-year CPI)
```

## 1. Design Philosophy · 设计思路

- **Tabular → 2D image**: a sliding window of `T` years × `K` indicators becomes a 2D grid, so convolution compares *neighbouring cells* and encodes **local contrast**.
- **Three-channel input** `(T, K, 3)`:
  - Channel 0 — level (standardized value)
  - Channel 1 — time contrast `z[t,k] − z[t−1,k]`
  - Channel 2 — indicator contrast `z[t,k] − z[t,k−1]`
- **CNN + LSTM**: CNN extracts local spatial features; LSTM models long-range temporal dependencies.
- **Pure NumPy**: forward/backward (Conv2D, MaxPool, LSTM, Dense, Adam) implemented from scratch — lightweight and dependency-free.

## 2. Mathematical Foundations · 数学基础

- **Min–max normalization** — positive indicators `(x−min)/(max−min)`; negative reversed `(max−x)/(max−min)`.
- **Entropy weight method** (熵值法): `p_ij = x'_ij / Σ x'_ij`, `e_j = −k Σ p_ij ln p_ij`, `d_j = 1 − e_j`, `w_j = d_j / Σ d_j`, `CPI = Σ w_j x'_j`.
- **Z-score standardization** for network input (stats from training split only).
- **Conv2D** cross-correlation, **ReLU / sigmoid / tanh** activations, **MaxPool(1×2)**.
- **LSTM gates** (forget / input / output / candidate).
- **MSE loss** + **Adam** optimizer (β₁=0.9, β₂=0.999) with bias correction and LR decay.

## 3. Statistical Methods · 统计方式

- **Entropy weighting** — objective weights from information entropy (higher dispersion → lower entropy → higher weight).
- **80/20 train/val split** with leakage-free normalization statistics.
- **Metrics** — MSE (loss), MAE (interpretable error), R² (explained variance).

## 4. Algorithm Patterns · 算法模式

- **Sliding-window supervised learning** (`seq_len=5`): window of 5 years → next-year CPI.
- **2D-CNN feature extraction** → **LSTM sequence modeling** → **Dense regression**.
- **Backpropagation** incl. **BPTT** through the LSTM and transposed-conv / max-unpool gradients.
- **Regularization**: inverted dropout (0.15), LR decay, per-epoch checkpointing + best-model selection, reproducible seeding (`--seed`).

## Architecture · 架构

```
Input (T,K,3)
  → Conv2D(8, 3×3) + ReLU + MaxPool(1×2)
  → Conv2D(16, 3×3) + ReLU + MaxPool(1×2)
  → reshape (T, 32)
  → LSTM(24)
  → Dropout(0.15) → Dense(16) + ReLU → Dense(1)
```

## Files · 文件

| File · 文件 | Purpose · 用途 |
|---|---|
| `fetch_real_data.py` | Fetch real NBS indicators via akshare · 抓取真实统计局数据 |
| `generate_data.py` | Synthetic data generator (fallback) · 合成数据生成器 |
| `train_cnn.py` | 2D-CNN + LSTM trainer (pure NumPy) · 训练器 |
| `predict.py` | Next-year CPI prediction & ranking · 预测与排名 |
| `data/*.csv` | Indicator data, CPI, predictions · 数据与预测结果 |
| `models/*.h5` | Trained best model · 训练好的模型 |
| `report.html` | Bilingual technical report · 双语技术报告 |

## Quick Start · 快速开始

```bash
# 1. Setup (Windows .bat) — creates venv + installs numpy/pandas
setup.bat

# 2. Train
run_training.bat          # or: python train_cnn.py --epochs 80 --seq-len 5

# 3. Predict next year
run_prediction.bat        # or: python predict.py
```

Requirements: Python 3.8+, `numpy`, `pandas` (and `akshare` for real data fetching).

## License

Provided for study and demonstration. Data source: National Bureau of Statistics of China (国家统计局).
