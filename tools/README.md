# tools/ — Python ML side (training, quantization, simulator, exporters)

All Python commands run inside the project virtualenv:

```bash
source .venv/bin/activate        # created on Day 1 (see DAY1_REPORT.md)
python tools/train.py            # Stage 1: dataset + MLP/1D-CNN training (FP32)
python tools/quantize.py         # Stage 2 (Day 2)
python tools/export_model.py     # Stage 3 (Day 2)
python tools/simulate.py         # Stage 2/3 (Day 2) — golden-reference simulator
```

Pinned dependencies for the sprint live in `requirements.txt` (repo root).

## Day 1 status

| Script | Status |
|---|---|
| `train.py` | **Day 1** — synthetic dataset, MLP + 1D-CNN, FP32, saves `models/fp32/*.pt` + split |
| `quantize.py`, `export_model.py`, `simulate.py` | Day 2 |

## Conventions

- Deterministic: every script seeds RNGs (`torch.manual_seed`, `np.random.seed`) —
  re-running produces identical artifacts (guide: reproducibility check).
- Artifacts under `models/` are **generated** and may be regenerated; never hand-edit.
- The train/val/test split is saved to disk and reused by quantized-accuracy tests —
  never re-split.
