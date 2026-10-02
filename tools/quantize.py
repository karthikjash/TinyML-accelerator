#!/usr/bin/env python3
"""
quantize.py — Stage 2 (Day 2): INT8 post-training quantization per CONTRACT_1.

Reads  : models/fp32/*.pt + split.npz + norm.npz
Writes : models/quant/<model>/
           graph.json    layer graph + all quant params (basis of model.json)
           tensors.npz   int8 weights, int32 biases (canonical dtypes)
         models/fp32/X_norm.npy  (cached normalized inputs, shared by simulate.py)

Deterministic: calibration uses the first 200 TRAIN samples of the saved split.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # import tinyml + train as modules

from tinyml import contract_math as cm  # noqa: E402
import train as train_mod               # noqa: E402  (module import does NOT run main)

ROOT = os.path.dirname(HERE)
FP32 = os.path.join(ROOT, "models", "fp32")
OUT = os.path.join(ROOT, "models", "quant")
N_CALIB = 200
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Layer graph construction (from Day-1 state dicts; mirrors tools/train.py)
# ---------------------------------------------------------------------------

def mlp_graph(sd):
    return [
        dict(name="fc1", type="fc", weight="fc1_w", bias="fc1_b",
             weight_t=sd["net.1.weight"].numpy().T,      # [in,out] per Contract 2
             bias_t=sd["net.1.bias"].numpy(), activation="relu"),
        dict(name="fc2", type="fc", weight="fc2_w", bias="fc2_b",
             weight_t=sd["net.3.weight"].numpy().T,
             bias_t=sd["net.3.bias"].numpy(), activation="relu"),
        dict(name="fc3", type="fc", weight="fc3_w", bias="fc3_b",
             weight_t=sd["net.5.weight"].numpy().T,
             bias_t=sd["net.5.bias"].numpy(), activation="none"),
        dict(name="sm", type="softmax"),
    ]


def cnn_graph(sd):
    return [
        dict(name="conv1", type="conv1d", weight="conv1_w", bias="conv1_b",
             weight_t=sd["conv.0.weight"].permute(1, 0, 2).numpy(),  # [in,out,K]
             bias_t=sd["conv.0.bias"].numpy(),
             kernel=5, stride=1, pad="same", activation="relu"),
        dict(name="pool1", type="maxpool1d", kernel=2, stride=2),
        dict(name="conv2", type="conv1d", weight="conv2_w", bias="conv2_b",
             weight_t=sd["conv.3.weight"].permute(1, 0, 2).numpy(),
             bias_t=sd["conv.3.bias"].numpy(),
             kernel=3, stride=1, pad="same", activation="relu"),
        dict(name="pool2", type="maxpool1d", kernel=2, stride=2),
        dict(name="flat", type="flatten"),
        dict(name="fc", type="fc", weight="fc_w", bias="fc_b",
             weight_t=sd["fc.weight"].numpy().T,
             bias_t=sd["fc.bias"].numpy(), activation="none"),
        dict(name="sm", type="softmax"),
    ]


def load_normalized_inputs():
    """Rebuild the dataset deterministically and normalize with saved stats."""
    split = np.load(os.path.join(FP32, "split.npz"))
    norm = np.load(os.path.join(FP32, "norm.npz"))
    X, y = train_mod.make_dataset()
    # train.py permutes (X, y) with rng(7) BEFORE splitting — replicate exactly,
    # otherwise split.npz indices/labels refer to a different sample order.
    perm = np.random.default_rng(7).permutation(len(X))
    X, y = X[perm], y[perm]
    Xn = ((X - norm["mean"]) / norm["std"]).astype(np.float32)
    np.save(os.path.join(FP32, "X_norm.npy"), Xn)
    return Xn, split


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_model(model_name, graph_builder, Xn, split):
    print(f"== quantizing {model_name} ==")
    sd = torch.load(os.path.join(FP32, f"{model_name}.pt"), weights_only=True)
    layers = graph_builder(sd)

    is_cnn = any(L["type"] == "conv1d" for L in layers)
    calib_idx = split["train"][:N_CALIB]
    calib = Xn[calib_idx]                             # [N,128,3]
    if not is_cnn:
        calib = calib.reshape(len(calib_idx), -1)     # [N,384] for MLP

    # ---- calibration: per-layer FP32 output min/max over N_CALIB samples ----
    mins = [None] * len(layers)
    maxs = [None] * len(layers)
    for i in range(len(calib)):
        cur = calib[i]  # single sample: [128,3] CNN / [384] MLP — 2-D layer ops
        for li, L in enumerate(layers):
            t = L["type"]
            if t == "fc":
                cur = cm.fp32_fc(cur, L["weight_t"], L["bias_t"],
                                 relu=(L["activation"] == "relu"))
            elif t == "conv1d":
                cur = cm.fp32_conv1d(cur, L["weight_t"], L["bias_t"], L["pad"],
                                     relu=(L["activation"] == "relu"))
            elif t == "maxpool1d":
                cur = cm.maxpool1d_int8(cur, L["kernel"])  # pure max — dtype-safe on FP32
            elif t == "flatten":
                cur = cur.reshape(1, -1)
            elif t == "softmax":
                continue  # output range fixed [0,1] by definition
            lo, hi = float(cur.min()), float(cur.max())
            mins[li] = lo if mins[li] is None else min(mins[li], lo)
            maxs[li] = hi if maxs[li] is None else max(maxs[li], hi)

    in_scale, in_zp = cm.compute_act_quant_params(float(calib.min()), float(calib.max()))
    print(f"  input: scale={in_scale:.6f} zp={in_zp}")

    # ---- walk layers, assign scales, quantize weights/biases ----
    tensors = {}
    scale_in, zp_in = in_scale, in_zp
    graph_layers = []
    prev_out_shape = [1, 128, 3] if is_cnn else [1, 384]

    for li, L in enumerate(layers):
        t = L["type"]
        if t == "softmax":
            graph_layers.append(dict(
                name=L["name"], type="softmax",
                in_shape=list(prev_out_shape), out_shape=list(prev_out_shape),
                scale_in=scale_in, zp_in=zp_in,
                scale_out=1.0 / 255.0, zp_out=-128,
                lut_ref=f"{L['name']}_lut"))
            break

        scale_out, zp_out = cm.compute_act_quant_params(mins[li], maxs[li])
        if t == "fc":
            q_w, scale_w = cm.quantize_int8_symmetric(L["weight_t"])
            q_b = cm.quantize_bias_int32(L["bias_t"], scale_in, scale_w)
            # Flatten-order fix: PyTorch flattens conv output [C,T] channel-major,
            # our buffers keep [T,C] (time-major). Fold the permutation into the
            # FC weight rows so the runtime flatten stays trivial row-major.
            # new_row[c*T + t] = old_row[t*C + c]  (Contract 2, flatten note)
            if is_cnn and graph_layers and graph_layers[-1]["type"] == "flatten":
                flat_in = graph_layers[-1]["in_shape"]   # [1, T, C] before flatten
                T, C = flat_in[1], flat_in[2]
                idx = np.empty(T * C, dtype=np.int64)
                for c in range(C):
                    for tt in range(T):
                        idx[c * T + tt] = tt * C + c
                q_w = q_w[idx, :]
            tensors[L["weight"]], tensors[L["bias"]] = q_w, q_b
            in_f, out_f = q_w.shape
            g = dict(name=L["name"], type="fc", weight=L["weight"], bias=L["bias"],
                     weight_shape=[in_f, out_f],
                     in_shape=list(prev_out_shape), out_shape=[1, out_f],
                     scale_in=scale_in, scale_w=scale_w, scale_out=scale_out,
                     zp_in=zp_in, zp_out=zp_out, activation=L["activation"])
            prev_out_shape = [1, out_f]
        elif t == "conv1d":
            q_w, scale_w = cm.quantize_int8_symmetric(L["weight_t"])
            q_b = cm.quantize_bias_int32(L["bias_t"], scale_in, scale_w)
            tensors[L["weight"]], tensors[L["bias"]] = q_w, q_b
            c_in, c_out, K = q_w.shape
            T_out = prev_out_shape[1]
            g = dict(name=L["name"], type="conv1d", weight=L["weight"], bias=L["bias"],
                     weight_shape=[c_in, c_out, K],
                     kernel=K, stride=1, pad=L["pad"],
                     in_shape=list(prev_out_shape), out_shape=[1, T_out, c_out],
                     scale_in=scale_in, scale_w=scale_w, scale_out=scale_out,
                     zp_in=zp_in, zp_out=zp_out, activation=L["activation"])
            prev_out_shape = [1, T_out, c_out]
        elif t == "maxpool1d":
            K = L["kernel"]
            g = dict(name=L["name"], type="maxpool1d", kernel=K, stride=L["stride"],
                     in_shape=list(prev_out_shape),
                     out_shape=[1, prev_out_shape[1] // K, prev_out_shape[2]])
            prev_out_shape = g["out_shape"]
        elif t == "flatten":
            flat = int(np.prod(prev_out_shape[1:]))
            g = dict(name=L["name"], type="flatten",
                     in_shape=list(prev_out_shape), out_shape=[1, flat])
            prev_out_shape = [1, flat]
        else:
            raise ValueError(t)

        graph_layers.append(g)
        scale_in, zp_in = scale_out, zp_out

    graph = dict(
        format_version="0.1", model_name=model_name, quant_spec="CONTRACT_1 v0.1",
        input=dict(shape=[1, 128, 3] if is_cnn else [1, 384],
                   scale=in_scale, zero_point=in_zp),
        output=dict(shape=[1, 4], scale=1.0 / 255.0, zero_point=-128),
        layers=graph_layers)

    out_dir = os.path.join(OUT, model_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "graph.json"), "w") as f:
        json.dump(graph, f, indent=2)
    np.savez(os.path.join(out_dir, "tensors.npz"), **tensors)

    n_bytes = sum(int(v.nbytes) for v in tensors.values())
    print(f"  layers={len(graph_layers)}  weight+bias bytes={n_bytes}")
    return graph


def main():
    Xn, split = load_normalized_inputs()
    print(f"normalized inputs cached: X_norm.npy {Xn.shape}")
    quantize_model("mlp_demo", mlp_graph, Xn, split)
    quantize_model("cnn_demo", cnn_graph, Xn, split)
    print("Stage 2 part 1 (quantize): DONE — bundles in models/quant/")


if __name__ == "__main__":
    main()
