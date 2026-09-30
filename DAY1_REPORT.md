# DAY 1 REPORT — Stage 0 + Stage 1 Complete ✅

**Date:** 2026-09-30 · **Repo:** github.com/karthikjash/TinyML-accelerator (branch `main`)
**Sprint:** 4-day TinyML framework build (guide: `TinyML_4Day_Guide.pdf`)
**Gates today:** GATE 0 ✅ (host part) · GATE 1 ✅ — **on schedule for Day 2**

---

## 1. What was done (in order)

### Stage 0 — Setup & Contracts (GATE 0)

| Item | Status |
|---|---|
| Python virtualenv `.venv/` created (PEP 668 blocks system pip on Ubuntu 24.04) | ✅ |
| PyTorch (CPU) + NumPy installed and pinned in `requirements.txt` | ✅ |
| Host C toolchain verified: gcc 13.3, CMake 3.28 (see §4 for versions + why) | ✅ |
| Repo scaffold created: `tools/ firmware/ models/ tests/ docs/ scripts/` | ✅ |
| **CONTRACT 1** — INT8 requantize spec frozen (`docs/CONTRACT_1_REQUANTIZE_SPEC.md`) | ✅ |
| **CONTRACT 2** — model.json schema frozen (`docs/CONTRACT_2_MODEL_JSON_SCHEMA.md`) | ✅ |
| **CONTRACT 3** — golden-vector format + test protocol frozen (`docs/CONTRACT_3_GOLDEN_VECTORS.md`) | ✅ |
| Platform HAL created: `firmware/src/platform/` (host + ESP32 backends) | ✅ |
| Fixed-width type contract: `firmware/src/types.h` (q7_t/q31_t/q63_t) | ✅ |
| Host build + `hello_test.c` (13 checks: types, saturation cast, round-half-to-even, MAC) | ✅ |
| `scripts/check_all.sh` — one command runs all gates | ✅ |

**ESP-IDF note:** not installable in this build environment — the ESP32 firmware project
(`firmware/esp32/`) is prepared as Day 3 work **on your machine** (install guide §0.1).
Host-first strategy means nothing is blocked; the exact same C kernels are compiled
on-device in Day 3.

### Stage 1 — Dataset + Model Training (GATE 1)

`tools/train.py` — fully deterministic (fixed seeds), single command retrains everything.

| Item | Result |
|---|---|
| Dataset | synthetic 3-axis gestures, 4 classes, length 128, 1600 samples, sensor noise injected |
| Split | train 1120 / val 240 / test 240 — **saved to disk** (`models/fp32/split.npz`), reused Day 2 |
| Normalization | per-feature mean/std from train split only → `models/fp32/norm.npz` |
| **MLP** | Flatten→FC128→ReLU→FC64→ReLU→FC4 · **test_acc = 100.0%** · 57,796 params |
| **1D-CNN** | Conv1D(16,k5)+ReLU+Pool→Conv1D(32,k3)+ReLU+Pool→FC4 · **test_acc = 100.0%** · 5,924 params (~7.7 KB INT8) |
| Artifacts | `models/fp32/mlp_demo.pt`, `cnn_demo.pt`, `metrics.json` |
| Re-train command | `source .venv/bin/activate && python tools/train.py` |

> Note: the MLP's INT8 size (~75 KB) is larger than the guide's ~15 KB sketch because
> FC128 takes the full 384-length flattened input. It still fits ESP32 RAM easily;
> a slimmer MLP is a trivial re-run if wanted. Gate criterion was accuracy, which passed.

---

## 2. Gate evidence (actual output)

```text
GATE 0 (host part):
  pass sizeof q7_t == 1 ... pass rint -1.5 == -2 ... pass q63 MAC = 32258
  ALL HELLO TESTS PASSED (host toolchain verified)
  → proves: C11 build with -O3 -fsigned-char, int8/int32/int64 layout,
    two's-complement saturation cast, round-half-to-even FP default

GATE 1:
  mlp_demo: test_acc=1.0000 (target >= 0.95) -> PASS
  cnn_demo: test_acc=1.0000 (target >= 0.95) -> PASS
```

---

## 3. Key decisions & why (these bind Days 2–4)

| Decision | Why |
|---|---|
| **3 contracts frozen before any kernel code** | bit-exactness is only possible if Python and C implement the *same* arithmetic; the contracts are that definition (guide: "never debug two unknowns at once") |
| **Requantize = double multiply + rint (not TFLite M0/shift) for v0.1** | IEEE double + `rint` is deterministic across Python/host-C/ESP32; M0/shift is a documented backlog item required only before CMSIS-NN integration |
| **Softmax = 256-entry LUT** (exporter precomputes exp for all int8 inputs) | `exp()` differs by 1 ulp between glibc and ESP32 newlib → LUT removes libm from the device entirely |
| **ReLU zeroes at `zp_out`, not 0** | asymmetric activations: real 0.0 maps to zero-point, not int 0 (Contract 1 §3) |
| **Two-buffer dispatcher (no in-place, no overlap in v0.1)** | makes verification trivial; buffer-lifetime planner stays a clean backlog item |
| **Conv pad value = `zp_in`** | padding means "real zero", which is `zp_in` in quantized domain — classic silent-accuracy bug avoided by contract |
| **Golden vectors: JSON interchange + generated C headers** | JSON is portable/diffable; headers mean **zero parsing on ESP32**; SHA-256 manifest prevents stale-vector accidents |
| **Split + normalization saved to disk** | Day 2 quantized accuracy must be measured on the *same* test set — re-splitting would fake the numbers |

## 4. Versions used (and why)

| Tool | Version | Why |
|---|---|---|
| Python | 3.12.3 (system) | distro default; venv isolates packages |
| PyTorch | **2.14.1+cpu** | CPU wheel: no CUDA needed for tiny models; +cpu avoids ~2 GB GPU deps |
| NumPy | **2.5.2** | simulator/export math; pinned for reproducibility |
| gcc | 13.3.0 | host kernel tests; `-fsigned-char` matches ESP32 char behavior |
| CMake | 3.28.3 | ≥3.16 required; host test runner |
| ESP-IDF | **not installed here** | Day 3 on your machine (v5.x) — see Next Steps |
| Git remote | karthikjash/TinyML-accelerator | agreed project home |

## 5. Issues hit today & how they were fixed (logged for the debugging map)

1. **PEP 668** blocked `pip install` system-wide → created `.venv/` (standard practice anyway).
2. **Double transpose** crashed the CNN (layout applied twice) → layout ownership moved into `CNN1D.forward()` only.
3. **CMake include path** (`CMAKE_SOURCE_DIR` ≠ repo root with `-S tests/host`) → resolved relative to `CMAKE_CURRENT_SOURCE_DIR`.
4. **Missing `PLATFORM_HOST` define** → added to host CMake; the #error guard stays as a safety net.
5. **stdio-after-use in HAL header** → include moved above first use (GCC 14+ hard-errors on implicit decls).

## 6. Repo state after Day 1

```text
docs/        CONTRACT_1_REQUANTIZE_SPEC.md · CONTRACT_2_MODEL_JSON_SCHEMA.md · CONTRACT_3_GOLDEN_VECTORS.md
firmware/src/types.h · platform/{platform.h, platform_host.h, platform_esp32.h}
tools/       train.py (Stage 1) · README.md
tests/host/  CMakeLists.txt · hello_test.c
scripts/     check_all.sh
models/fp32/ mlp_demo.pt · cnn_demo.pt · split.npz · norm.npz · metrics.json   (generated)
requirements.txt · .gitignore · TinyML_4Day_Guide.pdf (+html source)
```

## 7. Next process — Day 2 checklist (Stage 2 + Stage 3)

1. `tools/quantize.py` — PTQ per Contract 1: calibrate (~200 samples) → scales/zps → int8 weights, int32 biases. *(Stage 2)*
2. `tools/simulate.py` — NumPy **bit-exact simulator** (the golden reference): FC/Conv1D int32 MAC + double requantize + rint-half-even + saturate; ReLU at zp; LUT softmax. *(Stage 2)*
3. Quantized-accuracy check on the saved test split — **gate: ≤2% drop from FP32**. *(Stage 2 gate)*
4. `tools/export_model.py` — emit `models/<name>/model.json` + `model_data.c/.h` + golden vectors (JSON + generated `vectors.h/.c`) + SHA-256 manifest. *(Stage 3)*
5. Exporter **round-trip test** (re-parse generated C, re-simulate, compare). *(Stage 3 gate)*
6. `scripts/check_all.sh` grows: golden-vector host tests wired for the day-3 kernels' interface.
7. Commit + push + **DAY2_REPORT.md** in the same format as this file.

**Definition of done Day 2:** simulator deterministic · quantized accuracy within 2% · exporter round-trip passes · golden vectors exist for every layer of both models.

## 8. Reproduce Day 1 from clean checkout

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/train.py      # GATE 1
./scripts/check_all.sh               # GATE 0 (host part)
```
