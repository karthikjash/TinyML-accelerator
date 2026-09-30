"""
contract_math.py — the ONE implementation of CONTRACT_1 arithmetic.

Every quantization/simulation step in the project (quantize.py, simulate.py,
export_model.py, and — by derivation — all C kernels) must match this file
bit-for-bit. Reference: docs/CONTRACT_1_REQUANTIZE_SPEC.md

Key rules implemented here (contract section numbers):
  §2  requantize = double multiply + round-half-to-even + saturate + zp
  §3  ReLU zeroes at zp_out (asymmetric activations), maxpool over int8
  §4  softmax via 256-entry exp LUT, fixed left-to-right sum order
  §5  conv1d pads with zp_in, stride 1, weights [in_ch][out_ch][kernel]
  §7  accumulators clamped to int32 range
"""
import numpy as np

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


# ---------------------------------------------------------------------------
# Core CONTRACT_1 primitives
# ---------------------------------------------------------------------------

def rint_half_to_even(x):
    """CONTRACT_1 §2: round-half-to-even, elementwise (np.rint)."""
    return np.rint(x)


def requantize(acc_i64, requant_scale, zp_out):
    """CONTRACT_1 §2 exact requantize path for one output tensor.

    acc_i64: int64 accumulator (pre-clamp)
    requant_scale: float64 (scale_in * scale_w) / scale_out
    zp_out: int
    """
    acc_i32 = np.clip(acc_i64, INT32_MIN, INT32_MAX)
    out_real = acc_i32.astype(np.float64) * np.float64(requant_scale)
    out_q = rint_half_to_even(out_real).astype(np.int64) + np.int64(zp_out)
    return np.clip(out_q, INT8_MIN, INT8_MAX).astype(np.int8)


def quantize_int8_symmetric(w_float):
    """CONTRACT_1 §1: weights, symmetric (zp = 0).

    Returns (q_int8, scale_w_float64).
    """
    amax = float(np.max(np.abs(w_float)))
    if amax == 0.0:
        return np.zeros_like(w_float, dtype=np.int8), 1.0
    scale = amax / 127.0
    q = np.clip(rint_half_to_even(w_float.astype(np.float64) / scale),
                INT8_MIN, INT8_MAX).astype(np.int8)
    return q, scale


def quantize_int8_asymmetric(x_float, scale, zp):
    """CONTRACT_1 §1: activations, asymmetric. x must already be normalized."""
    q = np.clip(rint_half_to_even(x_float.astype(np.float64) / scale) + np.int64(zp),
                INT8_MIN, INT8_MAX).astype(np.int8)
    return q


def compute_act_quant_params(min_val, max_val):
    """CONTRACT_1 §2.1: scale = (max-min)/255 ; zp = round(-min/scale) - 128."""
    min_val = float(min_val)
    max_val = float(max_val)
    if max_val <= min_val:
        max_val = min_val + 1e-8
    scale = (max_val - min_val) / 255.0
    zp = int(np.clip(rint_half_to_even(np.float64(-min_val / scale)) - 128,
                     INT8_MIN, INT8_MAX))
    return scale, zp


def quantize_bias_int32(b_float, scale_in, scale_w):
    """CONTRACT_1 §1: b_q = round(b / (scale_in * scale_w)), int32."""
    denom = float(scale_in) * float(scale_w)
    q = rint_half_to_even(np.asarray(b_float, dtype=np.float64) / denom)
    return np.clip(q, INT32_MIN, INT32_MAX).astype(np.int32)


# ---------------------------------------------------------------------------
# Layer simulators (exact operation order the C kernels must copy)
# ---------------------------------------------------------------------------

def mac_fc(q_in, q_w, q_b, zp_in):
    """FC MAC with activation zero-point subtraction (CONTRACT_1 §2, v0.1.1).

    acc = sum_k w[k] * (x_q[k] - zp_in) + b, int64, clamped int32.
    q_in:  [in_features] int8 (raw quantized values, zp NOT yet removed)
    q_w:   [in_features, out_features] int8 (Contract 2 weight_shape)
    q_b:   [out_features] int32
    """
    x_true = q_in.astype(np.int64) - np.int64(zp_in)
    acc = (q_w.astype(np.int64) * x_true[:, None]).sum(axis=0)
    acc = acc + q_b.astype(np.int64)
    return np.clip(acc, INT32_MIN, INT32_MAX)


def mac_conv1d(q_in, q_w, q_b, kernel, pad_mode, zp_in):
    """Conv1d MAC, stride 1, with zp subtraction (CONTRACT_1 §2/§5, v0.1.1).

    q_in: [T, in_ch] int8 (raw quantized), q_w: [in_ch, out_ch, K] int8.
    Pads with zp_in ("same") — after subtraction a pad contributes exactly 0.
    """
    T, C_in = q_in.shape
    _, C_out, K = q_w.shape
    if pad_mode == "same":
        pad_total = K - 1
        left = pad_total // 2
        right = pad_total - left
        padded = np.full((T + pad_total, C_in), zp_in, dtype=np.int8)
        padded[left:left + T, :] = q_in
    else:  # valid
        padded = q_in
    Tp = padded.shape[0]
    T_out = Tp - K + 1
    x_true = padded.astype(np.int64) - np.int64(zp_in)
    acc = np.zeros((T_out, C_out), dtype=np.int64)
    for k in range(K):
        window = x_true[k:k + T_out, :]                           # [T_out, C_in]
        wk = q_w[:, :, k].astype(np.int64)                        # [C_in, C_out]
        acc += window @ wk
    acc = acc + q_b.astype(np.int64)[None, :]
    return np.clip(acc, INT32_MIN, INT32_MAX)


def relu_int8(q, zp_out):
    """CONTRACT_1 §3: ReLU in quantized domain zeroes at zp_out."""
    return np.maximum(q, np.int8(zp_out))


def maxpool1d_int8(q, kernel):
    """CONTRACT_1 §3: plain int8 max over non-overlapping windows."""
    T, C = q.shape
    T_out = T // kernel
    return q[: T_out * kernel].reshape(T_out, kernel, C).max(axis=1)


def softmax_int8_lut(q_in, lut, scale_out, zp_out):
    """CONTRACT_1 §4: exp LUT lookup -> fixed left-to-right sum -> int8 probs.

    lut: float64[256], lut[q + 128] = exp((q - zp_in) * scale_in)
    """
    e = lut[q_in.astype(np.int32) + 128]                # float64 lookups
    total = 0.0
    for v in e:                                          # fixed order sum
        total += float(v)
    p = e / total                                        # float64 division
    out = np.clip(rint_half_to_even(p / scale_out) + np.int64(zp_out),
                  INT8_MIN, INT8_MAX).astype(np.int8)
    return out


def build_softmax_lut(zp_in, scale_in):
    """256-entry LUT per CONTRACT_1 §4: LUT[q+128] = exp((q - zp_in) * scale_in)."""
    q = np.arange(INT8_MIN, INT8_MAX + 1, dtype=np.int64)
    return np.exp((q - np.int64(zp_in)) * np.float64(scale_in))


# ---------------------------------------------------------------------------
# FP32 (reference) layer ops — used for calibration and accuracy checks
# ---------------------------------------------------------------------------

def fp32_fc(x, w, b, relu=False):
    out = x @ w + b
    return np.maximum(out, 0.0) if relu else out


def fp32_conv1d(x, w, b, pad_mode, relu=False):
    """x: [T, C_in], w: [C_in, C_out, K] (torch layout transposed), stride 1."""
    T, C_in = x.shape
    _, C_out, K = w.shape
    if pad_mode == "same":
        pad_total = K - 1
        left = pad_total // 2
        right = pad_total - left
        padded = np.pad(x, ((left, right), (0, 0)))
    else:
        padded = x
    T_out = padded.shape[0] - K + 1
    out = np.zeros((T_out, C_out), dtype=np.float64)
    for k in range(K):
        out += padded[k:k + T_out, :] @ w[:, :, k]
    out += b[None, :]
    return np.maximum(out, 0.0) if relu else out
