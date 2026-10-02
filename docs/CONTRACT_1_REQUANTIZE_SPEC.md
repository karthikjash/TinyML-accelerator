# CONTRACT 1 — INT8 Quantization & Requantize Spec (v0.1.1, FROZEN 2026-09-30)

> v0.1.1 change (2026-09-30, found by simulator debugging): §2 now REQUIRES the
> activation zero-point subtraction inside the MAC (see §2 note).
>
> This document is **normative**. The Python simulator (`tools/simulate.py`, Day 2)
> and every C kernel (`firmware/src/kernels/`, Day 2–3) must implement exactly the
> arithmetic defined here. A kernel that deviates is **wrong**, even if its accuracy
> is fine. All bit-exactness tests derive from this document.

## 1. Number formats

| Value | Format | Notes |
|---|---|---|
| Weights | int8, **symmetric** (zp_w = 0) | `q = clip(round(w / scale_w), -128, 127)` |
| Biases | int32 | `q = round(b / (scale_in * scale_w))` |
| Activations (all layers) | int8, **asymmetric** | `q = clip(round(x / scale) + zp, -128, 127)` |
| Accumulators | int32 (computed in int64 internally, stored/saturated int32) | never overflows for v0.1 model sizes |
| Requantize constant | float64 `requant_scale = (scale_in * scale_w) / scale_out` | one double multiply per output element |

Per-tensor scales (one scale/zero-point per whole tensor). No per-channel scales in v0.1.

## 2. Fully-connected / Conv1d layer — exact arithmetic

For one output element:

```text
acc_i64 = sum_over_k( w_q[k] * (x_q[k] - zp_in) )   // int64 math; ZERO-POINT SUBTRACTED
acc_i32 = acc_i64 + b_q                              // b_q already carries bias scale
out_real = (double)acc_i32 * requant_scale
out_q   = rint_half_to_even(out_real)                // round-half-to-even
out_q   = out_q + zp_out
out_q   = saturate_int8(out_q)                       // clip to [-128, 127]
```

**The zero-point subtraction is mandatory** (v0.1.1). With asymmetric activations,
`q_x = x/scale + zp_in`; the MAC must operate on the true value `q_x - zp_in`.
Equivalent formulation (fold into bias, chosen for the C kernels):
`b_q_eff = b_q - zp_in * sum_k(w_q[k])`, precomputed at export time.
Conv1d padding uses `zp_in`, which then contributes exactly 0 to every MAC.

**Rounding rule: round-half-to-even** (IEEE default; Python `np.rint`, C `rint()` under
default FP mode). Both sides MUST be compiled/run with default rounding mode.

**Why a double multiply instead of TFLite's integer M0/shift:** v0.1 optimizes for
bit-exactness across Python / host C / ESP32 with minimal implementation risk. IEEE-754
double multiply + `rint` is deterministic on all three. The integer M0/shift path
(CMSIS-NN-compatible) is a documented backlog item — required before CMSIS-NN
integration, not before this sprint.

## 3. Activation functions

| Op | Rule |
|---|---|
| ReLU | applied after requantize: `out_q = max(out_q, zp_out)` (zero in real domain = zp_out, **not** 0) |
| MaxPool1d | plain int8 max over window; stride = kernel (2,2 in v0.1); no scales involved |
| Flatten | pure memory re-interpretation; no arithmetic |
| Softmax | see §4 |

## 4. Softmax — LUT-based (no libm on device)

`exp()` may differ by 1 ulp between host glibc and ESP32 newlib → **forbidden**.
Deterministic algorithm instead:

1. Exporter precomputes a **256-entry float64 LUT**: `LUT[q] = exp((q - zp_in) * scale_in)`
   for every possible int8 input value `q` (input to softmax is the FC logits tensor).
2. C/Python softmax: for each input element, `e_i = LUT[q_i]` (pure table lookup).
3. `sum = Σ e_i` (float64, fixed left-to-right order).
4. `p_i = e_i / sum` (float64).
5. Requantize to int8 output: `out_q = saturate(rint_half_to_even(p_i / scale_out) + zp_out)`.

The LUT (2 KB) lives in `model_data.c`. Sum order is fixed left-to-right on both sides.

## 5. Conv1d specifics

- Padding ("same"/"valid"): padding value in the **quantized domain is `zp_in`**
  (zero in real domain), never 0.
- Stride 1 only in v0.1. Kernel sizes 3 and 5 supported.
- Weights layout: `[out_channels][in_channels][kernel]`, row-major, in model_data.c.

## 6. Input normalization contract

Normalization (per-feature mean/std from training) is applied in the **float domain
before quantization**. Golden vectors carry **already-normalized, already-quantized**
inputs. The device never sees raw sensor values in v0.1 (raw-path = case-study backlog).

## 7. Saturation & overflow

- All int8 results saturate to [-128, 127] after rounding and zero-point add.
- Accumulator assumed int32-safe for v0.1 models (largest row: FC 1024×4 → max |acc|
  ≈ 1024·128·127 ≈ 1.7e7 ≪ 2^31). The exporter MUST assert this per layer.

## 8. What is explicitly out of spec for v0.1

Per-channel quantization, M0/shift requantize, fused conv+relu bias tricks, int16
activations, dynamic fixed-point. Each is a backlog item; changes to this spec require
a version bump and regeneration of ALL golden vectors.
