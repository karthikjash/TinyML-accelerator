#!/usr/bin/env python3
"""Debug: FP32 vs INT8-simulator layer-by-layer on a few train samples."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tinyml import contract_math as cm  # noqa: E402
import simulate as sim                  # noqa: E402

ROOT = os.path.dirname(HERE)
FP32 = os.path.join(ROOT, "models", "fp32")

Xn = np.load(os.path.join(FP32, "X_norm.npy"))
split = np.load(os.path.join(FP32, "split.npz"))
y = split["labels"]

for model_name in ("mlp_demo",):
    graph, tensors = sim.load_bundle(model_name)
    layers = graph["layers"]
    print(f"=== {model_name} ===")
    print("input:", graph["input"])

    for si in range(3):
        i = int(split["train"][si])
        x = Xn[i].reshape(1, -1)  # [1,384]
        label = int(y[i])
        print(f"\n-- sample {si} (idx {i}, label {label}) --")

        # FP32 pass with the same graph structure
        cur = x[0]
        fp32_outs = []
        for L in layers:
            t = L["type"]
            if t == "fc":
                # dequantize weights for a reference fp32 matmul
                q_w = tensors[L["weight"]].astype(np.float64) * L["scale_w"]
                q_b = tensors[L["bias"]].astype(np.float64) * (L["scale_in"] * L["scale_w"])
                cur = cm.fp32_fc(cur.reshape(1, -1), q_w, q_b[0],
                                 relu=(L["activation"] == "relu"))[0]
            elif t == "softmax":
                e = np.exp(cur - cur.max())
                cur = e / e.sum()
            fp32_outs.append(cur.copy())

        # INT8 pass with trace
        q_in = sim.quantize_input(x, graph["input"]["scale"], graph["input"]["zero_point"])
        q_in = q_in.reshape(graph["input"]["shape"])
        trace = []
        q_out = sim.simulate_model(graph, tensors, q_in, trace)

        for (name, qin, qout), fpo in zip(trace, fp32_outs):
            print(f"  layer {name}: int8 out[:8] = {qout.reshape(-1)[:8]}")
        print(f"  fp32 logits[:8]      = {fp32_outs[-2].reshape(-1)[:8].round(3)}")
        print(f"  fp32 probs[:8]       = {fp32_outs[-1].reshape(-1)[:4].round(3)}")
        print(f"  int8 probs (deq)     = {(q_out.reshape(-1).astype(np.float64)+128)/255}")
        print(f"  pred int8={int(np.argmax(q_out))}  pred fp32={int(np.argmax(fp32_outs[-1]))}")

    # input dequant sanity
    print("\ninput quant sanity: first 6 normalized vs dequantized:")
    q_in = sim.quantize_input(Xn[0].reshape(1, -1), graph["input"]["scale"], graph["input"]["zero_point"])
    deq = (q_in.astype(np.float64) - graph["input"]["zero_point"]) * graph["input"]["scale"]
    print("  ", Xn[0].reshape(-1)[:6].round(4))
    print("  ", deq.reshape(-1)[:6].round(4))
