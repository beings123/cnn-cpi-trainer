# -*- coding: utf-8 -*-
"""
2D-CNN + LSTM trainer for Common Prosperity Index (CPI) prediction
===================================================================
升级要点（相比原一维 CNN）：
  1. 表格 -> 二维数据：把滑动窗口 (年份 x 指标) 展成二维网格，
     并派生 3 个通道，显式编码「局部对比」信息。
  2. Conv2D 二维局部处理 + 局部对比：3x3 卷积核同时在「年份」和「指标」
     两个方向上滑动，比较相邻单元格（局部对比）。
  3. 增加 LSTM 增强：CNN 提取二维局部特征后，沿时间轴送入 LSTM，
     建模长程时序依赖，再做回归预测。

Features:
  - Terminal ASCII loss curve (live update)
  - Auto checkpoint per epoch
  - Auto-select best model (lowest validation loss)
  - Graceful Ctrl+C interrupt
  - Pure NumPy implementation (no TF/PyTorch needed)

Usage:
  python train_cnn.py
  python train_cnn.py --epochs 100 --seq-len 6
  python train_cnn.py --light
"""

import os
import sys
import argparse
import signal
import time
import numpy as np
import pandas as pd

# ============================================================
# Config
# ============================================================
parser = argparse.ArgumentParser(description='2D-CNN + LSTM CPI Prediction Trainer')
parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--seq-len', type=int, default=5, help='Sequence window length (years)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--light', action='store_true', help='Light mode (faster)')
parser.add_argument('--data', type=str, default='data/common_prosperity_data.csv', help='Data path')
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed. Default: RANDOM each run (genuine run-to-run variance). '
                         'Pass an integer to reproduce one specific run.')
args = parser.parse_args()

if args.light:
    args.epochs = 50
    args.batch_size = 32
    args.seq_len = 4

# ---------------------------------------------------------------------------
# Random seed policy
#   - Default: draw a fresh seed from OS entropy every run, so weight init,
#     data shuffling and dropout masks all differ -> results vary naturally.
#   - With --seed N: deterministic, so a single run can be reproduced exactly.
# ---------------------------------------------------------------------------
RUN_SEED = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'little')
np.random.seed(RUN_SEED)

CHECKPOINT_DIR = 'checkpoints'
BEST_MODEL_PATH = 'models/best_cnn_lstm_model.h5'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global state for graceful interrupt
training_interrupted = False
best_val_loss = float('inf')
best_epoch = -1


def signal_handler(sig, frame):
    global training_interrupted
    training_interrupted = True
    print("\n\n[!] Interrupt detected. Stopping training gracefully...")


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# Terminal ASCII Chart
# ============================================================
class TerminalChart:
    """Draw live loss curve in terminal using ASCII characters"""

    def __init__(self, width=60, height=16):
        self.width = width
        self.height = height
        self.train_losses = []
        self.val_losses = []

    def update(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def render(self):
        h = self.height
        w = self.width
        n = len(self.train_losses)

        if n == 0:
            return ""

        all_vals = self.train_losses + self.val_losses
        y_max = max(all_vals) * 1.1
        y_min = min(all_vals) * 0.9
        if y_max == y_min:
            y_max += 0.001

        if n <= w:
            step = 1
            display_n = n
        else:
            step = n / w
            display_n = w

        canvas = [[' ' for _ in range(w)] for _ in range(h)]

        for row in range(0, h, 4):
            for col in range(w):
                if canvas[row][col] == ' ':
                    canvas[row][col] = '.'

        y_labels = []
        for i in range(5):
            val = y_max - (y_max - y_min) * i / 4
            y_labels.append(f"{val:.4f}")

        def get_row(val):
            ratio = (val - y_min) / (y_max - y_min)
            row = int((1 - ratio) * (h - 1))
            return max(0, min(h - 1, row))

        for col in range(display_n):
            idx = min(int(col * step), n - 1)

            t_row = get_row(self.train_losses[idx])
            canvas[t_row][col] = '*'

            v_row = get_row(self.val_losses[idx])
            if canvas[v_row][col] == ' ':
                canvas[v_row][col] = 'o'
            elif canvas[v_row][col] == '*':
                canvas[v_row][col] = '8'

        lines = []
        label_width = 8

        for row in range(h):
            if row % 4 == 0:
                label_idx = row // 4
                label = y_labels[label_idx] if label_idx < len(y_labels) else ''
            else:
                label = ''
            lines.append(f"{label:>{label_width}} |" + ''.join(canvas[row]))

        lines.append(' ' * label_width + ' +' + '-' * w)
        lines.append(' ' * label_width + '  ' + f"Epoch 1 -> Epoch {n}".center(w))
        lines.append(' ' * label_width + '  ' + "* Train Loss   o Val Loss   8 Overlap")

        return '\n'.join(lines)


# ============================================================
# 2D Data Construction  (table -> 2D image)
# ============================================================
FEATURE_COLS = ['GDP_per_capita', 'Disposable_income', 'Urbanization_rate',
                'Education_ratio', 'Social_security',
                'Urban_rural_income_ratio', 'Environment_index',
                'Infrastructure_index']


def build_2d_channels(window, feature_mean, feature_std):
    """
    Convert a raw (T, K) window into a (T, K, 3) 2D multi-channel "image".

    Channel 0: level             -- standardized indicator level
    Channel 1: time contrast     -- x[t,k] - x[t-1,k]，相邻年份局部对比
    Channel 2: indicator contrast-- x[t,k] - x[t,k-1]，相邻指标局部对比

    window: (T, K) raw values
    returns: (T, K, 3)
    """
    T, K = window.shape
    z = (window - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)

    ch_time = np.zeros_like(z)
    ch_time[1:, :] = z[1:, :] - z[:-1, :]      # 纵向（时间）局部对比

    ch_ind = np.zeros_like(z)
    ch_ind[:, 1:] = z[:, 1:] - z[:, :-1]       # 横向（指标）局部对比

    return np.stack([z, ch_time, ch_ind], axis=-1)   # (T, K, 3)


def load_data(csv_path, seq_len=5):
    """Load table, normalize, and build 2D multi-channel inputs"""
    print("[Data] Loading dataset and reshaping table into 2D grids...")
    df = pd.read_csv(csv_path)
    provinces = df['Province'].unique()

    windows = []
    targets = []

    for prov in provinces:
        prov_data = df[df['Province'] == prov].sort_values('Year')
        values = prov_data[FEATURE_COLS].values
        cpi = prov_data['CPI'].values
        for i in range(len(values) - seq_len):
            windows.append(values[i:i + seq_len])
            targets.append(cpi[i + seq_len])

    windows = np.array(windows)      # (N, T, K)
    targets = np.array(targets)      # (N,)

    # Shuffle & split 80/20
    indices = np.random.permutation(len(windows))
    split = int(0.8 * len(windows))
    train_idx, val_idx = indices[:split], indices[split:]

    # Normalization stats from TRAINING data only (no leakage)
    train_flat = windows[train_idx].reshape(-1, len(FEATURE_COLS))
    feature_mean = np.mean(train_flat, axis=0)
    feature_std = np.std(train_flat, axis=0) + 1e-8

    target_mean = np.mean(targets[train_idx])
    target_std = np.std(targets[train_idx]) + 1e-8

    # Build 2D channels
    X_all = np.stack([build_2d_channels(w, feature_mean, feature_std) for w in windows], axis=0)
    y_all = (targets - target_mean) / target_std

    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    norm_params = {
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'target_mean': target_mean,
        'target_std': target_std,
    }

    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
    print(f"  2D input shape:(T={seq_len}, K={len(FEATURE_COLS)}, C=3)  [level + time-contrast + indicator-contrast]")
    print(f"  Target mean:   {target_mean:.4f}")
    print(f"  Target std:    {target_std:.4f}")

    return X_train, y_train, X_val, y_val, norm_params


# ============================================================
# Low-level ops
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x):
    return np.maximum(0, x)


def pad_same_2d(x, p=1):
    """Zero-pad the height and width dims of (B,H,W,C)."""
    return np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')


def conv2d_forward(x_pad, W):
    """Cross-correlation conv2d. x_pad: (B,H,W,Cin), W: (kh,kw,Cin,Cout) -> (B,H-kh+1,W-kw+1,Cout)"""
    kh, kw, Cin, Cout = W.shape
    B, H, Wd, _ = x_pad.shape
    oh, ow = H - kh + 1, Wd - kw + 1
    out = np.zeros((B, oh, ow, Cout))
    for i in range(kh):
        for j in range(kw):
            patch = x_pad[:, i:i + oh, j:j + ow, :]          # (B,oh,ow,Cin)
            out += np.einsum('bhwc,cd->bhwd', patch, W[i, j])
    return out


def conv2d_backward(dout, x_pad, W):
    """Returns dW, dx_pad"""
    kh, kw, Cin, Cout = W.shape
    B, H, Wd, _ = x_pad.shape
    oh, ow = H - kh + 1, Wd - kw + 1
    dW = np.zeros_like(W)
    dx_pad = np.zeros_like(x_pad)
    for i in range(kh):
        for j in range(kw):
            patch = x_pad[:, i:i + oh, j:j + ow, :]
            dW[i, j] += np.einsum('bhwc,bhwd->cd', patch, dout)
            dx_pad[:, i:i + oh, j:j + ow, :] += np.einsum('bhwd,cd->bhwc', dout, W[i, j])
    return dW, dx_pad


def maxpool_w_forward(x, size=2):
    """Max-pool over the width (indicator) axis only; preserves the time axis."""
    B, H, Wd, C = x.shape
    ow = Wd // size
    xr = x[:, :, :ow * size, :].reshape(B, H, ow, size, C)
    idx = np.argmax(xr, axis=3)                              # (B,H,ow,C)
    out = np.take_along_axis(xr, idx[..., None, :], axis=3)[..., 0, :]
    return out, idx


def maxpool_w_backward(dout, idx, x_shape, size=2):
    B, H, Wd, C = x_shape
    ow = Wd // size
    dxr = np.zeros((B, H, ow, size, C))
    np.put_along_axis(dxr, idx[..., None, :], dout[..., None, :], axis=3)
    dx = np.zeros(x_shape)
    dx[:, :, :ow * size, :] = dxr.reshape(B, H, ow * size, C)
    return dx


# ============================================================
# 2D-CNN + LSTM model (pure NumPy)
# ============================================================
class CNNLSTM:
    """
    Architecture:
        Input (T, K, 3)
        -> Conv2D(8, 3x3, same) -> ReLU -> MaxPool(1x2 over indicators)   # 二维局部处理 + 局部对比
        -> Conv2D(16,3x3, same) -> ReLU -> MaxPool(1x2 over indicators)
        -> reshape to sequence (T, 2*16)
        -> LSTM(24)                                                       # 增强长程时序依赖
        -> Dropout -> Dense(16) -> ReLU -> Dense(1)
    """

    def __init__(self, seq_len, n_features, n_channels=3, lr=0.001, dropout_rate=0.15,
                 nf1=8, nf2=16, lstm_hidden=24, dense_hidden=16, seed=None):
        self.lr = lr
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate

        self.nf1, self.nf2 = nf1, nf2
        self.H = lstm_hidden
        self.dense_hidden = dense_hidden

        # seed=None -> weight init differs every run (no longer pinned to 0)
        rng = np.random.RandomState(seed)

        # Conv2D #1: 3x3, same padding
        self.W1 = rng.randn(3, 3, n_channels, nf1) * np.sqrt(2.0 / (3 * 3 * n_channels))
        self.b1 = np.zeros(nf1)

        # pooling over indicator axis halves K twice
        k1 = max(1, n_features // 2)
        k2 = max(1, k1 // 2)
        self.k2 = k2
        self.seq_feat = k2 * nf2

        # Conv2D #2: 3x3, same padding
        self.W2 = rng.randn(3, 3, nf1, nf2) * np.sqrt(2.0 / (3 * 3 * nf1))
        self.b2 = np.zeros(nf2)

        # LSTM: input dim = seq_feat, hidden = H, gates = 4H
        d = self.seq_feat + self.H
        self.Wl = rng.randn(d, 4 * self.H) * np.sqrt(2.0 / d)
        self.bl = np.zeros(4 * self.H)

        # Dense head
        self.W3 = rng.randn(self.H, dense_hidden) * np.sqrt(2.0 / self.H)
        self.b3 = np.zeros(dense_hidden)
        self.W4 = rng.randn(dense_hidden, 1) * np.sqrt(2.0 / dense_hidden)
        self.b4 = np.zeros(1)

        self._init_optimizer()
        self.cache = {}
        self.training = True

    # ---------------- optimizer ----------------
    def _param_names(self):
        return ['W1', 'b1', 'W2', 'b2', 'Wl', 'bl', 'W3', 'b3', 'W4', 'b4']

    def _init_optimizer(self):
        self.opt_m = {n: np.zeros_like(getattr(self, n)) for n in self._param_names()}
        self.opt_v = {n: np.zeros_like(getattr(self, n)) for n in self._param_names()}
        self.opt_t = 0

    # ---------------- forward ----------------
    def forward(self, X):
        """X: (B, T, K, C) -> (B,) predictions (normalized scale)"""
        B, T, K, C = X.shape

        # --- Conv block 1 (2D local processing + comparison) ---
        xp1 = pad_same_2d(X, 1)
        c1 = conv2d_forward(xp1, self.W1) + self.b1
        a1 = relu(c1)                                        # (B,T,K,nf1)
        p1, idx1 = maxpool_w_forward(a1, 2)                  # (B,T,K/2,nf1)

        # --- Conv block 2 ---
        xp2 = pad_same_2d(p1, 1)
        c2 = conv2d_forward(xp2, self.W2) + self.b2
        a2 = relu(c2)                                        # (B,T,K/2,nf2)
        p2, idx2 = maxpool_w_forward(a2, 2)                  # (B,T,K/4,nf2)

        # --- reshape feature map into a time sequence for LSTM ---
        Bf, Tf, k2, nf2 = p2.shape
        seq = p2.reshape(Bf, Tf, k2 * nf2)                   # (B,T,seq_feat)

        # --- LSTM over time ---
        h_seq, lstm_cache = self._lstm_forward(seq)
        h_last = h_seq[:, -1, :]                             # (B,H)

        # --- Dropout ---
        if self.training:
            mask = (np.random.rand(*h_last.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            hd = h_last * mask
        else:
            mask = None
            hd = h_last

        # --- Dense head ---
        d1 = hd @ self.W3 + self.b3
        r1 = relu(d1)
        out = (r1 @ self.W4 + self.b4).reshape(-1)

        self.cache = dict(
            X=X, xp1=xp1, c1=c1, a1=a1, idx1=idx1,
            xp2=xp2, c2=c2, a2=a2, idx2=idx2, p2=p2,
            seq=seq, lstm_cache=lstm_cache,
            h_last=h_last, mask=mask, hd=hd, d1=d1, r1=r1,
        )
        return out

    def _lstm_forward(self, seq):
        B, T, _ = seq.shape
        H = self.H
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = []
        hs = []
        for t in range(T):
            concat = np.concatenate([h, seq[:, t, :]], axis=1)      # (B, H+D)
            z = concat @ self.Wl + self.bl                          # (B, 4H)
            zf, zi, zo, zg = np.split(z, 4, axis=1)
            f = sigmoid(zf)
            i = sigmoid(zi)
            o = sigmoid(zo)
            g = np.tanh(zg)
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            cache.append((h, c, f, i, o, g, c_new, tc, concat))
            hs.append(h_new)
            h, c = h_new, c_new
        return np.stack(hs, axis=1), cache

    # ---------------- backward ----------------
    def backward(self, y_pred, y_true):
        B = len(y_true)
        c = self.cache

        # MSE gradient
        d_out = (2.0 / B) * (y_pred - y_true).reshape(-1, 1)     # (B,1)

        # Dense head
        dW4 = c['r1'].T @ d_out
        db4 = d_out.sum(axis=0)
        d_r1 = d_out @ self.W4.T
        d_d1 = d_r1 * (c['d1'] > 0)
        dW3 = c['hd'].T @ d_d1
        db3 = d_d1.sum(axis=0)
        d_hd = d_d1 @ self.W3.T

        # Dropout
        d_h_last = d_hd * c['mask'] if c['mask'] is not None else d_hd

        # LSTM backward: gradient only flows from the final timestep output
        T = c['seq'].shape[1]
        dh_seq = np.zeros((B, T, self.H))
        dh_seq[:, -1, :] = d_h_last
        dWl, dbl, d_seq = self._lstm_backward(dh_seq)

        # reshape sequence grads back to feature map
        d_p2 = d_seq.reshape(c['p2'].shape)

        # Pool2 -> ReLU2 -> Conv2
        d_a2 = maxpool_w_backward(d_p2, c['idx2'], c['a2'].shape)
        d_c2 = d_a2 * (c['c2'] > 0)
        dW2, d_xp2 = conv2d_backward(d_c2, c['xp2'], self.W2)
        d_b2 = d_c2.sum(axis=(0, 1, 2))
        d_p1 = d_xp2[:, 1:-1, 1:-1, :]                            # crop same padding

        # Pool1 -> ReLU1 -> Conv1
        d_a1 = maxpool_w_backward(d_p1, c['idx1'], c['a1'].shape)
        d_c1 = d_a1 * (c['c1'] > 0)
        dW1, d_xp1 = conv2d_backward(d_c1, c['xp1'], self.W1)
        d_b1 = d_c1.sum(axis=(0, 1, 2))

        return {'W1': dW1, 'b1': d_b1, 'W2': dW2, 'b2': d_b2,
                'Wl': dWl, 'bl': dbl, 'W3': dW3, 'b3': db3,
                'W4': dW4, 'b4': db4}

    def _lstm_backward(self, dh_seq):
        """Backprop through time. Returns dWl, dbl, d_seq"""
        c = self.cache
        seq = c['seq']
        B, T, _ = seq.shape
        H = self.H

        dWl = np.zeros_like(self.Wl)
        dbl = np.zeros_like(self.bl)
        d_seq = np.zeros_like(seq)

        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))

        for t in reversed(range(T)):
            h_prev, c_prev, f, i, o, g, c_new, tc, concat = c['lstm_cache'][t]

            dh = dh_seq[:, t, :] + dh_next

            do = dh * tc
            dc = dh * o * (1 - tc ** 2) + dc_next
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            dzf = df * f * (1 - f)
            dzi = di * i * (1 - i)
            dzo = do * o * (1 - o)
            dzg = dg * (1 - g ** 2)
            dz = np.concatenate([dzf, dzi, dzo, dzg], axis=1)

            dWl += concat.T @ dz
            dbl += dz.sum(axis=0)

            d_concat = dz @ self.Wl.T
            dh_next = d_concat[:, :H]
            d_seq[:, t, :] = d_concat[:, H:]
            dc_next = dc_prev

        return dWl, dbl, d_seq

    # ---------------- optimizer step ----------------
    def update(self, grads):
        self.opt_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for n in self._param_names():
            g = grads[n]
            self.opt_m[n] = beta1 * self.opt_m[n] + (1 - beta1) * g
            self.opt_v[n] = beta2 * self.opt_v[n] + (1 - beta2) * (g ** 2)
            m_hat = self.opt_m[n] / (1 - beta1 ** self.opt_t)
            v_hat = self.opt_v[n] / (1 - beta2 ** self.opt_t)
            setattr(self, n, getattr(self, n) - self.lr * m_hat / (np.sqrt(v_hat) + eps))

    def num_params(self):
        return int(sum(getattr(self, n).size for n in self._param_names()))

    # ---------------- persistence ----------------
    def save(self, path, norm_params=None):
        data = {n: getattr(self, n) for n in self._param_names()}
        data.update({
            'seq_len': self.seq_len, 'n_features': self.n_features,
            'n_channels': self.n_channels, 'lr': self.lr,
            'dropout_rate': self.dropout_rate, 'nf1': self.nf1, 'nf2': self.nf2,
            'lstm_hidden': self.H, 'dense_hidden': self.dense_hidden,
            'opt_m': self.opt_m, 'opt_v': self.opt_v, 'opt_t': self.opt_t,
        })
        if norm_params is not None:
            data['norm_params'] = norm_params
        tmp = path + '.npy'
        np.save(tmp, data, allow_pickle=True)
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp, path)

    def load(self, path):
        load_path = path if os.path.exists(path) else path + '.npy'
        data = np.load(load_path, allow_pickle=True).item()
        for n in self._param_names():
            setattr(self, n, data[n])
        self.seq_len = data['seq_len']
        self.n_features = data['n_features']
        self.n_channels = data.get('n_channels', 3)
        self.lr = data.get('lr', self.lr)
        self.dropout_rate = data.get('dropout_rate', self.dropout_rate)
        self.nf1 = data.get('nf1', self.nf1)
        self.nf2 = data.get('nf2', self.nf2)
        self.H = data.get('lstm_hidden', self.H)
        self.dense_hidden = data.get('dense_hidden', self.dense_hidden)
        self.k2 = max(1, max(1, self.n_features // 2) // 2)
        self.seq_feat = self.k2 * self.nf2

        self.opt_m = data.get('opt_m', {n: np.zeros_like(getattr(self, n)) for n in self._param_names()})
        self.opt_v = data.get('opt_v', {n: np.zeros_like(getattr(self, n)) for n in self._param_names()})
        self.opt_t = data.get('opt_t', 0)
        self.training = True
        return data.get('norm_params', None)


# ============================================================
# Training
# ============================================================
def train():
    global best_val_loss, best_epoch, training_interrupted

    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 74)
    print("  2D-CNN + LSTM  Common Prosperity Index Predictor  (pure NumPy)")
    print("=" * 74)
    print(f"   Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  Seq Len: {args.seq_len}  |  LR: {args.lr}")
    print(f"   Run seed: {RUN_SEED}"
          + ("  (fixed via --seed)" if args.seed is not None else "  (random each run)"))
    print("=" * 74)
    print()

    X_train, y_train, X_val, y_val, norm_params = load_data(args.data, args.seq_len)
    print()

    print("[Model] Building 2D-CNN + LSTM ...")
    model = CNNLSTM(seq_len=args.seq_len, n_features=len(FEATURE_COLS),
                    n_channels=3, lr=args.lr, dropout_rate=0.15, seed=RUN_SEED)
    print(f"   Arch : Input(T={args.seq_len}, K={len(FEATURE_COLS)}, C=3)")
    print(f"          -> Conv2D({model.nf1},3x3,same)+ReLU+MaxPool(1x2)")
    print(f"          -> Conv2D({model.nf2},3x3,same)+ReLU+MaxPool(1x2)")
    print(f"          -> reshape (T, {model.seq_feat}) -> LSTM({model.H})")
    print(f"          -> Dropout -> Dense({model.dense_hidden}) -> Dense(1)")
    print(f"   Params: {model.num_params()} (ultra-lightweight, CPU-friendly)")
    print()

    chart = TerminalChart()
    n_train = len(X_train)
    n_batches = (n_train + args.batch_size - 1) // args.batch_size
    initial_lr = args.lr

    print("[Train] Starting... (Ctrl+C to interrupt, best model auto-saved)")
    print("-" * 74)
    time.sleep(1)

    for epoch in range(1, args.epochs + 1):
        if training_interrupted:
            break

        t0 = time.time()
        model.lr = initial_lr / (1.0 + 0.01 * (epoch - 1))    # LR decay
        model.training = True

        idx = np.random.permutation(n_train)
        X_sh, y_sh = X_train[idx], y_train[idx]

        loss_sum, n_seen = 0.0, 0
        for b in range(n_batches):
            s, e = b * args.batch_size, min((b + 1) * args.batch_size, n_train)
            Xb, yb = X_sh[s:e], y_sh[s:e]
            yp = model.forward(Xb)
            loss_sum += np.mean((yp - yb) ** 2) * len(yb)
            n_seen += len(yb)
            model.update(model.backward(yp, yb))

        train_loss = loss_sum / n_seen

        model.training = False
        val_pred = model.forward(X_val)
        val_loss = np.mean((val_pred - y_val) ** 2)

        # denormalized metrics
        tm, ts = norm_params['target_mean'], norm_params['target_std']
        yv = y_val * ts + tm
        pv = val_pred * ts + tm
        val_mae = np.mean(np.abs(pv - yv))
        val_r2 = 1 - np.sum((yv - pv) ** 2) / np.sum((yv - np.mean(yv)) ** 2)

        dt = time.time() - t0
        chart.update(train_loss, val_loss)

        model.save(os.path.join(CHECKPOINT_DIR, f'epoch{epoch:03d}_val{val_loss:.6f}.ckpt'), norm_params)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            model.save(BEST_MODEL_PATH, norm_params)
            marker = "  [NEW BEST!]"
        else:
            marker = ""

        if epoch > 1:
            print(f"\033[{chart.height + 6}A", end='')

        print(f"\r[Epoch {epoch:3d}/{args.epochs}]  Train: {train_loss:.6f}  |  "
              f"Val: {val_loss:.6f}  |  R2: {val_r2:.4f}  |  Time: {dt:.2f}s{marker}")
        print()
        print(chart.render())
        print()

        filled = int(50 * epoch / args.epochs)
        bar = '#' * filled + '-' * (50 - filled)
        print(f"   Progress: [{bar}] {epoch}/{args.epochs} ({epoch / args.epochs * 100:.1f}%)")
        print(f"   Best so far: Epoch {best_epoch}, Val Loss = {best_val_loss:.6f}")
        sys.stdout.flush()

    # -------- finalize --------
    print()
    print("=" * 74)
    print("  [DONE] Training complete!")
    print("=" * 74)
    print(f"  Best epoch : {best_epoch}")
    print(f"  Best val   : {best_val_loss:.8f}")
    print(f"  Model file : {BEST_MODEL_PATH}")
    print()

    final = CNNLSTM(seq_len=args.seq_len, n_features=len(FEATURE_COLS))
    final_norm = final.load(BEST_MODEL_PATH)
    final.training = False

    tm, ts = final_norm['target_mean'], final_norm['target_std']
    tr_p = final.forward(X_train) * ts + tm
    va_p = final.forward(X_val) * ts + tm
    tr_y = y_train * ts + tm
    va_y = y_val * ts + tm

    print("  Best model evaluation (original CPI scale):")
    print(f"    Train MSE: {np.mean((tr_p - tr_y) ** 2):.8f}")
    print(f"    Val MSE  : {np.mean((va_p - va_y) ** 2):.8f}")
    print(f"    Val MAE  : {np.mean(np.abs(va_p - va_y)):.8f}")
    print(f"    Val R2   : {1 - np.sum((va_y - va_p) ** 2) / np.sum((va_y - va_y.mean()) ** 2):.4f}")
    print()

    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.ckpt')]
    print(f"  Checkpoints: {len(ckpts)} files in {CHECKPOINT_DIR}/")
    print("=" * 74)

    return final, final_norm


if __name__ == '__main__':
    train()
