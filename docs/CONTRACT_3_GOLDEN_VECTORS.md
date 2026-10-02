# CONTRACT 3 — Golden-Vector Format & Test Protocol (v0.1, FROZEN 2026-09-30)

> Golden vectors are the single source of truth for correctness. The Python
> simulator (Contract 1 arithmetic) produces them; host C and ESP32 tests must
> reproduce them **bit-exactly** (tolerance 0).

## 1. File layout

```text
tests/golden/<model_name>/
├── manifest.json          # what this suite contains, hashes, quant params
├── L00_input.json         # the quantized model input
├── L01_fc1.json           # input + expected output of layer 1 (post-activation)
├── L02_relu1.json         # ...
├── ...
├── Llast_softmax.json
└── full_model.json        # model input → final output
```

- One file per layer, named `L<index padded 2>_<layer_name>.json` (names match
  `model.json` `layers[].name` exactly).
- `full_model.json` duplicates the chain end-to-end for dispatcher tests.

## 2. Per-layer vector schema

```json
{
  "format_version": "0.1",
  "model_name": "mlp_demo",
  "layer_index": 1,
  "layer_name": "fc1",
  "layer_type": "fc",
  "input":  { "data": [ -12, 5, ... ], "shape": [1, 128] },
  "output": { "data": [ 45, -3, ... ], "shape": [1, 64] },
  "tolerance": 0
}
```

Rules:
- `data` arrays are **flat, row-major int8** (values −128..127 as JSON ints).
- `shape` duplicates model.json for cross-validation.
- `tolerance` is 0 (bit-exact) in v0.1 — a test runner seeing tolerance > 0 must fail loudly.
- Softmax layer vectors additionally include the 256-entry `lut` used (float64 list),
  so C tests can verify LUT loading.

## 3. manifest.json

```json
{
  "format_version": "0.1",
  "model_name": "mlp_demo",
  "model_json_sha256": "...",
  "model_data_h_sha256": "...",
  "num_layers": 7,
  "layer_files": ["L00_input.json", "..."],
  "generator": "tools/simulate.py v0.1",
  "created": "2026-10-01T09:00:00Z"
}
```

Test runners MUST verify both SHA-256 hashes before running: golden vectors are only
valid against the exact model artifacts they were generated from. Stale vectors must
fail, not silently pass.

## 4. Test runners (who consumes vectors, and how)

| Runner | Location | Runs | Failure mode |
|---|---|---|---|
| Host kernel tests | `tests/host/test_kernels.c` (CMake) | per-layer: load vector, run ONE kernel, memcmp output | non-zero exit, prints first mismatching index |
| Host dispatcher test | `tests/host/test_full_model.c` | `full_model.json` through the whole runtime | same |
| ESP32 on-target tests | `firmware/main/golden_test_main.c` | vectors embedded at build; runs per-layer + full; prints PASS/FAIL per vector over UART | prints `GOLDEN_FAIL <layer> <idx>` and blinks LED |

All three runners share one tiny JSON-ish parser (vectors are simple enough to parse
with a fixed-field reader) — or, simpler and chosen for v0.1: **the exporter also emits
C header versions of the vectors** (`tests/golden/<model>/vectors.h` + `.c`) so C tests
link data directly and no JSON parsing exists on device at all. JSON files remain the
portable interchange + diff format; headers are generated from them in the same export
step, and manifest hashes cover both.

## 5. Regeneration rule

Golden vectors are **never hand-edited**. Any change to model, dataset, quant params
or Contract 1 → re-run `tools/run_day2_export.sh` (Day 2) → new vectors + new hashes
→ re-run all three runners. CI (or the Day-1 script `scripts/check_all.sh`) runs host
tests on every push.

## 6. Acceptance

A stage counts as verified only when its runner prints `ALL GOLDEN TESTS PASSED
(<n> vectors, tolerance 0)` for both models on host — and, from Day 3, on ESP32.
