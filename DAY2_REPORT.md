# DAY 2 REPORT — Stage 2 + Stage 3 Complete ✅

**Date:** 2026-09-30 · **Repo:** github.com/karthikjash/TinyML-accelerator (branch `main`)
**Sprint:** 4-day TinyML framework build · **Gates today:** GATE 2 ✅ · GATE 3 ✅ — **on schedule for Day 3**

---

## 1. What was done (in order)

| # | Item | Result |
|---|---|---|
| 1 | **Stage 2 — `tools/quantize.py`** (INT8 PTQ per Contract 1 §1–§2) | ✅ calibration on first 200 train samples → per-layer scales/zps → int8 weights, int32 biases → `models/quant/<name>/{graph.json,tensors.npz}` |
| 2 | **Stage 2 — `tools/simulate.py` + `tools/tinyml/contract_math.py`** (the **golden reference**) | ✅ bit-exact NumPy simulator: int8→int64 MAC → int32 → double requantize → rint-half-even → saturate → zp; LUT softmax |
| 3 | **Stage 2 gate — quantized accuracy** on the *saved Day-1 test split* | ✅ **MLP 100.0% (0.00% drop) · CNN 100.0% (0.00% drop)** |
| 4 | **Stage 3 — `tools/export_model.py`** | ✅ per model: `model.json` + `model_data.c/.h` + golden vectors (JSON **and** generated `vectors.h/.c`, zero parsing on device) + `manifest.json` with SHA-256 of every artifact |
| 5 | **Stage 3 gate — exporter round-trip** (re-parse generated C → re-simulate → compare) | ✅ weights bit-exact, outputs bit-exact for both models |
| 6 | `scripts/check_all.sh` extended → **one command now runs GATES 0→3** | ✅ `ALL DAY 1-2 GATES: PASS` |

Export artifacts live in `models/export/<model>/` (720 KB total; per-layer vectors for every layer of both models, 8 full-model vectors each).

## 2. Two real quantization bugs found and fixed today (this is why the simulator exists)

1. **Missing zero-point subtraction in the MAC** (Contract 1 upgraded **v0.1 → v0.1.1**).
   My first MACs computed `Σ w_q·x_q` — but with asymmetric activations the true value is
   `Σ w_q·(x_q − zp_in)`. With `zp_in = −128` on post-ReLU tensors each MAC carried a
   constant offset of `−128·Σw`, crushing accuracy to chance (~29%).
   **Fix:** zp subtracted inside the MAC; contract §2 rewritten with a v0.1.1 changelog
   entry; padding note added (pad value `zp_in` now contributes exactly 0).
2. **Flatten-order mismatch (CNN only).** PyTorch flattens conv output `[C,T]`
   (channel-major); our buffers are `[T,C]` (time-major). The CNN's FC head received
   permuted features → 20% accuracy.
   **Fix:** the permutation is folded into the FC weight **rows at export time**
   (`new_row[c·T+t] = old_row[t·C+c]`), so the runtime flatten stays trivial row-major.
   Documented in Contract 2 (v0.1.1). Same fix will be needed verbatim for the C kernels.

Debug artifacts kept: `tools/debug_layer_compare.py` (per-layer FP32-vs-INT8 diff used
to find bug #1) — the Day-3 equivalent tool for C kernels.

## 3. Versions used today

| Tool | Version | Note |
|---|---|---|
| Python / venv | 3.12.3 / `.venv` (unchanged) | |
| NumPy | 2.5.2 | simulator math; int64 MACs, float64 requantize |
| PyTorch | 2.14.1+cpu | only for loading state dicts |
| New files | `contract_math.py`, `simulate.py`, `quantize.py`, `export_model.py`, `debug_layer_compare.py` | all deterministic, seeded |

## 4. Key decisions & why

| Decision | Why |
|---|---|
| **One shared math module** (`tinyml/contract_math.py`) used by quantizer, simulator AND exporter | single source of truth — C kernels (Day 3) are written against this file, not against three divergent implementations |
| **Golden vectors as generated C headers** (plus JSON) | device tests link data directly — no JSON parser on ESP32, no byte-order/format risk; SHA-256 manifest catches stale vectors |
| **Calibration = first 200 train samples, seeded** | deterministic re-runs (reproducibility check); train-only (no test leakage) |
| **Fold flatten-permutation into weights, not the runtime** | keeps the v0.1 dispatcher trivially simple; the C runtime does a pure row-major flatten |
| **Round-trip test re-parses the generated C** | proves the *artifact*, not just the exporter's in-memory state — catches printf/format bugs before ESP32 testing |

## 5. Issues hit & fixed (for the debugging map)

1. `permute(2,1,0)` vs `permute(1,0,2)` on conv weights (torch `[out,in,K]` → contract `[in,out,K]`).
2. Batch-vs-single shape in calibration loop (2-D layer ops need single samples).
3. `mac_fc` broadcast axis wrong (`[None,:]` vs `[:,None]`) — found after the zp fix exposed it.
4. Dataset rebuild missing `rng(7)` permutation → labels misaligned (caught by
   preds==fp32-but-accuracy-29% — a textbook case for "verify against the reference
   before debugging the model").
5. Flatten-order (see §2.2).

## 6. Repo state after Day 2

```text
tools/            quantize.py · simulate.py · export_model.py · debug_layer_compare.py
tools/tinyml/     contract_math.py  (THE arithmetic reference)
models/quant/     mlp_demo/ cnn_demo/          (graph.json + tensors.npz)
models/export/    mlp_demo/ cnn_demo/          (model.json, model_data.c/.h, golden/, manifest.json)
models/fp32/      + X_norm.npy                       (cached normalized inputs, shared)
scripts/check_all.sh                                 (GATES 0→3 in one command)
```

## 7. Next process — Day 3 checklist (Stage 4: C kernels + ESP32)

1. `firmware/src/kernels/`: `requantize`, `fc_int8`, `conv1d_int8`, `relu`, `maxpool`, `softmax_lut`
   — written **directly against `contract_math.py`** (same operation order, incl. the
   zp-subtract and the flatten-folded FC weights).
2. Host golden tests: `tests/host/test_kernels.c` + `test_full_model.c` running the
   generated `vectors.h` (GATE 4a: bit-exact on host).
3. **On your machine:** install ESP-IDF v5.x, create `firmware/esp32/` project wrapping
   `firmware/src/`, flash, run the same golden tests on device (GATE 4b: bit-exact on ESP32).
   I can generate the full ESP-IDF project skeleton + build instructions for you to flash.
4. Naive vs lightly-optimized kernels (packed weights / unrolled GEMM) — same vectors must pass.
5. Benchmark harness (cycles via `esp_cpu_get_cycle_count()`, RAM, flash) → first numbers.
6. Commit + push + **DAY3_REPORT.md** (same format).

**Definition of done Day 3:** host kernels bit-exact on all golden vectors · ESP32
bit-exact on all golden vectors · first naive-vs-optimized cycle table for both models.

## 8. Reproduce Day 2 from clean checkout

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/train.py        # GATE 1
.venv/bin/python tools/quantize.py     # Stage 2
.venv/bin/python tools/simulate.py     # GATE 2
.venv/bin/python tools/export_model.py # Stage 3 + GATE 3
./scripts/check_all.sh                 # all gates in one command
```
