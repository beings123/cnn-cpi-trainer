# -*- coding: utf-8 -*-
"""
Predict next-year CPI for all 31 provinces using the best 2D-CNN + LSTM model.
Builds the same 2D multi-channel input as training.
"""
import os
import numpy as np
import pandas as pd

from train_cnn import CNNLSTM, FEATURE_COLS, build_2d_channels

DATA_PATH = 'data/common_prosperity_data.csv'
MODEL_PATH = 'models/best_cnn_lstm_model.h5'


def predict_next_year():
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model not found: {MODEL_PATH}")
        print("Please run train_cnn.py first to train a model.")
        return

    print("=" * 74)
    print("  Common Prosperity Index - Next Year Prediction  (2D-CNN + LSTM)")
    print("=" * 74)
    print()

    df = pd.read_csv(DATA_PATH)
    last_year = df['Year'].max()
    next_year = last_year + 1

    probe = np.load(MODEL_PATH, allow_pickle=True).item()
    seq_len = probe['seq_len']
    n_features = probe['n_features']

    model = CNNLSTM(seq_len=seq_len, n_features=n_features)
    norm_params = model.load(MODEL_PATH)
    model.training = False

    if norm_params is None:
        print("[Warning] No normalization params in model. Using default scaling.")
        norm_params = {
            'feature_mean': np.zeros(n_features),
            'feature_std': np.ones(n_features),
            'target_mean': 0.5,
            'target_std': 0.2,
        }

    fmean = norm_params['feature_mean']
    fstd = norm_params['feature_std']
    tmean = norm_params['target_mean']
    tstd = norm_params['target_std']

    print(f"  Data last year:  {last_year}")
    print(f"  Predict year:    {next_year}")
    print(f"  Window (T):      {seq_len} years")
    print(f"  Indicators (K):  {n_features}")
    print(f"  2D input:        (T={seq_len}, K={n_features}, C=3)")
    print()

    results = []
    for prov in df['Province'].unique():
        prov_data = df[df['Province'] == prov].sort_values('Year')
        window = prov_data[FEATURE_COLS].values[-seq_len:]
        last_cpi = prov_data['CPI'].values[-1]

        X2d = build_2d_channels(window, fmean, fstd)[None, ...]   # (1,T,K,3)
        pred = model.forward(X2d)[0] * tstd + tmean

        results.append({
            'Province': prov,
            f'CPI_{last_year}': last_cpi,
            f'CPI_{next_year}_pred': pred,
            'Change': pred - last_cpi,
            'Change_%': (pred - last_cpi) / last_cpi * 100,
        })

    res = pd.DataFrame(results).sort_values(f'CPI_{next_year}_pred', ascending=False).reset_index(drop=True)

    print(f"  Predicted CPI for {next_year} (ranked):")
    print("-" * 74)
    print(f"  {'Rank':<5}{'Province':<18}{'Current':>10}{'Predicted':>11}{'Change':>10}{'Change%':>10}")
    print("-" * 74)

    for i, row in res.iterrows():
        cs = f"+{row['Change']:.4f}" if row['Change'] >= 0 else f"{row['Change']:.4f}"
        ps = f"+{row['Change_%']:.2f}%" if row['Change_%'] >= 0 else f"{row['Change_%']:.2f}%"
        mark = '  [Top 3]' if i < 3 else ('  [Bottom 3]' if i >= len(res) - 3 else '')
        print(f"  {i+1:<5}{row['Province']:<18}{row[f'CPI_{last_year}']:>10.4f}"
              f"{row[f'CPI_{next_year}_pred']:>11.4f}{cs:>10}{ps:>10}{mark}")

    print("-" * 74)
    print()
    mean_cpi = res[f'CPI_{next_year}_pred'].mean()
    max_cpi = res[f'CPI_{next_year}_pred'].max()
    min_cpi = res[f'CPI_{next_year}_pred'].min()
    n_up = int((res['Change_%'] > 0).sum())
    n_dn = int((res['Change_%'] <= 0).sum())

    print(f"  National prediction mean: {mean_cpi:.4f}")
    print(f"  Highest: {res.iloc[0]['Province']} ({max_cpi:.4f})")
    print(f"  Lowest:  {res.iloc[-1]['Province']} ({min_cpi:.4f})")
    print(f"  Gap:     {max_cpi - min_cpi:.4f}")
    print(f"  Avg growth: {res['Change_%'].mean():+.2f}%")
    print(f"  Provinces increasing: {n_up}")
    print(f"  Provinces decreasing: {n_dn}")
    print()

    out = 'data/predictions.csv'
    res.to_csv(out, index=False)
    print(f"  Results saved to: {out}")
    print("=" * 74)


if __name__ == '__main__':
    predict_next_year()
