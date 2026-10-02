#!/usr/bin/env python3
"""
train.py — Stage 1 (Day 1): synthetic gesture dataset + FP32 MLP & 1D-CNN.

Deterministic by design: fixed seeds for numpy and torch, so re-running this
script reproduces identical models, split and metrics (guide: reproducibility).

Outputs (models/fp32/):
  mlp_demo.pt        best-val MLP state_dict
  cnn_demo.pt        best-val 1D-CNN state_dict
  split.npz          train/val/test indices + labels (REUSED on Day 2 for
                     quantized accuracy — never re-split)
  norm.npz           per-feature mean/std (C side must apply the same)
  metrics.json       FP32 test accuracies (reference row of the results table)

Model sizes are deliberately small (guide §1.2): MLP ~15 KB, CNN ~25 KB INT8.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "models", "fp32")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset: synthetic 3-axis "gesture" signals, 4 classes, length 128.
#   class 0: sine-ish sweep    class 1: square-ish pulse train
#   class 2: damped ramp       class 3: noise burst
# Sufficient class separation -> >95% test accuracy expected (GATE 1).
# ---------------------------------------------------------------------------
N_CLASSES = 4
SEQ_LEN = 128
N_SAMPLES = 1600  # 400 per class


def make_dataset():
    X = np.zeros((N_SAMPLES, SEQ_LEN, 3), dtype=np.float32)
    y = np.zeros((N_SAMPLES,), dtype=np.int64)
    t = np.linspace(0, 4 * np.pi, SEQ_LEN, dtype=np.float32)
    rng = np.random.default_rng(1234)
    for i in range(N_SAMPLES):
        c = i % N_CLASSES
        y[i] = c
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.8, 1.2)
        if c == 0:  # sweep across 3 axes with phase offsets
            X[i, :, 0] = amp * np.sin(t + phase)
            X[i, :, 1] = amp * np.sin(t + phase + 2.1)
            X[i, :, 2] = amp * np.sin(t + phase + 4.2)
        elif c == 1:  # pulse train
            duty = rng.uniform(0.3, 0.5)
            sq = (np.sin(t + phase) > (2 * duty - 1)).astype(np.float32)
            X[i, :, 0] = amp * sq
            X[i, :, 1] = -amp * sq
            X[i, :, 2] = amp * sq * np.sin(8 * t + phase)
        elif c == 2:  # damped ramp
            env = np.exp(-0.15 * np.arange(SEQ_LEN)).astype(np.float32)
            X[i, :, 0] = amp * env
            X[i, :, 1] = amp * env * np.sin(t + phase)
            X[i, :, 2] = -amp * env
        else:  # filtered noise burst
            burst = rng.normal(0, 0.35, (SEQ_LEN,)).astype(np.float32)
            kern = np.ones(5, dtype=np.float32) / 5.0
            sm = np.convolve(burst, kern, mode="same")
            X[i, :, 0] = burst
            X[i, :, 1] = sm
            X[i, :, 2] = np.roll(sm, 3) + rng.normal(0, 0.05, (SEQ_LEN,)).astype(np.float32)
        X[i] += rng.normal(0, 0.02, X[i].shape).astype(np.float32)  # sensor noise
    return X, y


class MLP(nn.Module):
    """Flatten -> FC128 -> ReLU -> FC64 -> ReLU -> FC n_classes  (~15 KB INT8)"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SEQ_LEN * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """Conv1D(16,k5)+ReLU+Pool -> Conv1D(32,k3)+ReLU+Pool -> FC  (~25 KB INT8)"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, padding=2),  # 'same'
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # 'same'
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        conv_out = 32 * (SEQ_LEN // 4)
        self.fc = nn.Linear(conv_out, N_CLASSES)

    def forward(self, x):
        # x: (N, L, C) -> (N, C, L) for Conv1d
        return self.fc(self.conv(x.transpose(1, 2)))


def train_model(model, Xtr, ytr, Xva, yva, name, epochs=15, batch=64, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None
    # CNN takes natural (N, L, C) layout; its forward() owns the channel swap.
    Xtr_t = torch.tensor(Xtr if name == "cnn_demo" else Xtr.reshape(len(Xtr), -1))
    ytr_t = torch.tensor(ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot_loss = 0.0
        for i in range(0, len(Xtr_t), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
        # validation
        model.eval()
        with torch.no_grad():
            Xva_t = torch.tensor(Xva if name == "cnn_demo" else Xva.reshape(len(Xva), -1))
            val_acc = (model(Xva_t).argmax(1).numpy() == yva).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [{name}] epoch {ep+1:2d}/{epochs}  loss {tot_loss/len(Xtr):.4f}  val_acc {val_acc:.4f}")
    model.load_state_dict(best_state)
    return best_val


def main():
    print("== Stage 1: generating synthetic gesture dataset ==")
    X, y = make_dataset()
    idx = np.random.default_rng(7).permutation(len(X))
    X, y = X[idx], y[idx]
    n_tr, n_va = int(0.7 * len(X)), int(0.15 * len(X))
    splits = {"train": idx[:n_tr], "val": idx[n_tr : n_tr + n_va], "test": idx[n_tr + n_va :]}
    print(f"  samples={len(X)}  shape={X.shape[1:]}  classes={N_CLASSES}")
    print(f"  split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # Normalization per feature (axis,channel) from TRAIN split only.
    Xtr = X[splits["train"]]
    mean = Xtr.reshape(-1, 3).mean(axis=0)
    std = Xtr.reshape(-1, 3).std(axis=0) + 1e-8
    Xn = ((X - mean) / std).astype(np.float32)
    print(f"  norm mean={mean.round(3)} std={std.round(3)}")

    Xtr_n, Xva_n, Xte_n = Xn[splits["train"]], Xn[splits["val"]], Xn[splits["test"]]
    ytr, yva, yte = y[splits["train"]], y[splits["val"]], y[splits["test"]]

    metrics = {}
    for name, ctor in (("mlp_demo", MLP), ("cnn_demo", CNN1D)):
        print(f"== training {name} ==")
        model = ctor()
        val_acc = train_model(model, Xtr_n, ytr, Xva_n, yva, name)
        model.eval()
        with torch.no_grad():
            Xte_t = torch.tensor(Xte_n if name == "cnn_demo" else Xte_n.reshape(len(Xte_n), -1))
            test_acc = float((model(Xte_t).argmax(1).numpy() == yte).mean())
        params = sum(p.numel() for p in model.parameters())
        int8_kb = (params + params // 3) / 1024.0  # weights int8 + bias ~int32/4
        print(f"  {name}: val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
              f"params={params} int8_size~{int8_kb:.1f} KB")
        metrics[name] = {"val_acc": val_acc, "test_acc": test_acc, "params": params}
        torch.save(model.state_dict(), os.path.join(OUT, f"{name}.pt"))

    np.savez_compressed(os.path.join(OUT, "split.npz"),
                        train=splits["train"], val=splits["val"], test=splits["test"],
                        labels=y)
    np.savez_compressed(os.path.join(OUT, "norm.npz"), mean=mean, std=std)
    metrics["dataset"] = {"n_samples": len(X), "seq_len": SEQ_LEN, "n_classes": N_CLASSES,
                          "split": {k: int(len(v)) for k, v in splits.items()}}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("== GATE 1 ==")
    ok = True
    for m in ("mlp_demo", "cnn_demo"):
        acc = metrics[m]["test_acc"]
        target = 0.95  # synthetic data target (guide)
        status = "PASS" if acc >= target else "FAIL"
        ok &= acc >= target
        print(f"  {m}: test_acc={acc:.4f} (target >= {target}) -> {status}")
    if not ok:
        sys.exit(1)
    print("GATE 1: PASS — models trained, split + norm + metrics saved to models/fp32/")


if __name__ == "__main__":
    main()
