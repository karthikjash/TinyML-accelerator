#!/usr/bin/env python3
"""
simulate.py — Stage 2 (Day 2): the NumPy BIT-EXACT simulator (golden reference).

Runs the quantized model exactly as the C kernels must (CONTRACT_1):
int8 MACs -> int32 -> double requantize -> rint-half-even -> saturate -> zp.
The export step (export_model.py) emits golden vectors from THIS simulator's
outputs, and every C test is judged against them with tolerance 0.

Usage:
  python tools/simulate.py                # quantized-accuracy gate on test split
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tinyml import contract_math as cm  # noqa: E402

ROOT = os.path.dirname(HERE)
QUANT = os.path.join(ROOT, "models", "quant")
FP32 = os.path.join(ROOT, "models", "fp32")


def load_bundle(model_name):
    with open(os.path.join(QUANT, model_name, "graph.json")) as f:
        graph = json.load(f)
    tensors = dict(np.load(os.path.join(QUANT, model_name, "tensors.npz")))
    return graph, tensors


def run_layer(L, q_in, tensors, trace=None):
    """Run one layer on int8 input exactly per CONTRACT_1. q_in: int8 array."""
    t = L["type"]
    if t == "fc":
        q_flat = q_in.reshape(-1)                       # [in_features]
        acc = cm.mac_fc(q_flat, tensors[L["weight"]], tensors[L["bias"]],
                        L["zp_in"])
        q_out = cm.requantize(acc, (L["scale_in"] * L["scale_w"]) / L["scale_out"],
                              L["zp_out"]).reshape(1, -1)
    elif t == "conv1d":
        q_2d = q_in.reshape(-1, q_in.shape[-1])         # [T, C]
        acc = cm.mac_conv1d(q_2d, tensors[L["weight"]], tensors[L["bias"]],
                            L["kernel"], L["pad"], L["zp_in"])
        q_out = cm.requantize(acc, (L["scale_in"] * L["scale_w"]) / L["scale_out"],
                              L["zp_out"])
    elif t == "maxpool1d":
        q_out = cm.maxpool1d_int8(q_in.reshape(-1, q_in.shape[-1]), L["kernel"])
    elif t == "flatten":
        q_out = q_in.reshape(1, -1)
    elif t == "relu":
        q_out = cm.relu_int8(q_in, L["zp_out"])
    elif t == "softmax":
        lut = cm.build_softmax_lut(L["zp_in"], L["scale_in"])
        q_out = cm.softmax_int8_lut(q_in.reshape(-1), lut,
                                    L["scale_out"], L["zp_out"]).reshape(1, -1)
    else:
        raise ValueError(t)

    if trace is not None:
        trace.append((L["name"], q_in.copy(), q_out.copy()))
    return q_out


def simulate_model(graph, tensors, q_input, trace=None):
    """Full-model int8 inference. q_input: int8 shaped graph['input']['shape']."""
    q = q_input
    for L in graph["layers"]:
        q = run_layer(L, q, tensors, trace)
    return q


def quantize_input(x_norm, scale, zp):
    return cm.quantize_int8_asymmetric(x_norm, scale, zp)


def main():
    metrics_fp32 = json.load(open(os.path.join(FP32, "metrics.json")))
    print("== Stage 2: quantized-accuracy gate (simulator vs FP32 reference) ==")
    report = {}
    ok = True
    for model_name in ("mlp_demo", "cnn_demo"):
        graph, tensors = load_bundle(model_name)
        Xn = np.load(os.path.join(FP32, "X_norm.npy"))
        split = np.load(os.path.join(FP32, "split.npz"))
        y = split["labels"] if "labels" in split else None
        test_idx = split["test"]
        y_test = np.array([y[i] for i in test_idx]) if y is not None else None

        inp = graph["input"]
        correct = 0
        for n, i in enumerate(test_idx):
            q_in = quantize_input(Xn[i], inp["scale"], inp["zero_point"])
            q_in = q_in.reshape(graph["input"]["shape"])
            q_out = simulate_model(graph, tensors, q_in)
            pred = int(np.argmax(q_out.astype(np.int32).reshape(-1)))
            if y_test is not None and pred == int(y_test[n]):
                correct += 1
        acc = correct / len(test_idx)
        fp32_acc = metrics_fp32[model_name]["test_acc"]
        drop = (fp32_acc - acc) * 100.0
        status = "PASS" if drop <= 2.0 else "FAIL"
        ok &= drop <= 2.0
        report[model_name] = {"quant_acc": acc, "fp32_acc": fp32_acc, "drop_pct": drop}
        print(f"  {model_name}: INT8 acc={acc:.4f}  FP32 acc={fp32_acc:.4f}  "
              f"drop={drop:.2f}%  -> {status}")

    print("GATE 2 (quantized accuracy): " + ("PASS" if ok else "FAIL"))
    with open(os.path.join(QUANT, "quant_accuracy.json"), "w") as f:
        json.dump(report, f, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
