# CONTRACT 2 — model.json Schema (v0.1, FROZEN 2026-09-30)

> Emitted by `tools/export_model.py` (Day 2). Consumed by the C dispatcher and
> test runners. **No C code may hardcode shapes, scales or sizes** — everything
> comes from this file (compiled into the firmware via the generated
> `model_config.h` / `model_data.h`).

## Top-level object

```json
{
  "format_version": "0.1",
  "model_name": "mlp_demo",
  "quant_spec": "CONTRACT_1 v0.1",
  "input":  { "shape": [1, 128], "scale": 0.003921, "zero_point": -128 },
  "output": { "shape": [1, 4],   "scale": 0.003921, "zero_point": -128 },
  "arena_size_bytes": 2048,
  "layers": [ ... ordered array, index = execution order ... ]
}
```

## Layer objects (only the fields shown exist; unknown fields are errors)

**fc**
```json
{ "type": "fc", "name": "fc1",
  "in_shape": [1, 128], "out_shape": [1, 64],
  "weight": "fc1_w", "bias": "fc1_b",
  "weight_shape": [128, 64],
  "scale_in": 0.0039, "scale_w": 0.0011, "scale_out": 0.0042,
  "zp_in": -128, "zp_out": -128,
  "activation": "relu" }
```

**conv1d**
```json
{ "type": "conv1d", "name": "conv1",
  "in_shape": [1, 128, 1], "out_shape": [1, 128, 16],
  "weight": "conv1_w", "bias": "conv1_b",
  "weight_shape": [1, 16, 5],
  "kernel": 5, "stride": 1, "pad": "same",
  "scale_in": 0.0039, "scale_w": 0.0011, "scale_out": 0.0042,
  "zp_in": -128, "zp_out": -128,
  "activation": "relu" }
```

**relu / softmax / maxpool1d / flatten**
```json
{ "type": "relu",      "name": "relu2", "in_shape": [1, 64], "out_shape": [1, 64], "zp_in": -128, "zp_out": -128 }
{ "type": "softmax",   "name": "sm",    "in_shape": [1, 4],  "out_shape": [1, 4],
  "scale_in": 0.0042, "zp_in": -128, "scale_out": 0.003921, "zp_out": -128, "lut_ref": "softmax_lut" }
{ "type": "maxpool1d", "name": "pool1", "in_shape": [1, 128, 16], "out_shape": [1, 64, 16], "kernel": 2, "stride": 2 }
{ "type": "flatten",   "name": "flat",  "in_shape": [1, 64, 16],  "out_shape": [1, 1024] }
```

## Field rules

| Field | Rule |
|---|---|
| `format_version` | must equal `"0.1"`; C dispatcher rejects anything else |
| `layers[].activation` | `"relu"`, `"none"` (fc/conv1d only) — activation is fused into the fc/conv call, NOT a separate layer entry |
| `weight` / `bias` / `lut_ref` | symbol names that MUST exist in `model_data.h` (see Contract 3 export) |
| `weight_shape` | fc: `[in_features, out_features]` (PyTorch linear weight transposed at export; **after a flatten following conv1d, FC weight rows are pre-permuted at export so runtime flatten stays row-major over [T,C] buffers** — v0.1.1); conv1d: `[in_ch, out_ch, kernel]` |
| scales | float64 JSON numbers; zero_points int |
| `arena_size_bytes` | = sum of the two largest consecutive intermediate buffers (double-buffer scheme, v0.1); exporter computes and asserts it |
| `in_shape`/`out_shape` | batch dim is always element 0 and MUST be 1 on device (single-inference benchmarking); exporters may emit batch > 1 for host tests only |

## Memory / buffer rules (v0.1 double-buffer dispatcher)

- Dispatcher allocates exactly two intermediate buffers sized to the max intermediate
  tensor; layer n reads buf[n%2], writes buf[(n+1)%2] (flatten is free; in-place ops
  are forbidden in v0.1 to keep verification trivial).
- The exporter asserts every layer's output fits and records `arena_size_bytes`.

## Validation

`tools/validate_model_json.py` (Day 2) checks: schema fields, shapes chaining
(prev out_shape == next in_shape), scale/zp presence, weight symbol references,
arena math. The C side re-validates shapes at boot and hard-faults (assert) on
mismatch — a model/config mismatch must never pass silently.
