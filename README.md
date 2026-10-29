# TinyML Inference Framework for NXP MCXN236

A model-driven, integer-only inference runtime for compact neural networks on the **NXP FRDM-MCXN236** (Arm Cortex-M33). This repository includes a reproducible Python model pipeline, exported INT8 MLP and 1D-CNN models, a portable C inference runtime, host correctness tests, and an MCXN236 verification application.

## Project at a glance

- Trains compact MLP and 1D-CNN classifiers on a deterministic synthetic sensor-signal dataset.
- Applies post-training INT8 quantization and exports model parameters and metadata for the C runtime.
- Runs inference with integer tensors, a statically supplied arena, and no heap allocation in the inference path.
- Checks C runtime outputs against checked-in golden vectors and compares optimized and reference execution modes.

The checked-in training results report 100% test accuracy for both FP32 demo models. Quantized evaluation reports 100% accuracy for both models, with no measured drop on the project's held-out split. These figures describe the synthetic dataset and do not establish real-world sensor performance.

## Architecture

```text
Python training → INT8 quantization → model export (C arrays + JSON metadata)
                                              ↓
Host tests ← C runtime ← model dispatcher + kernels + caller-provided arena
                                              ↓
                              MCXN236 verification application
```

The runtime and model definitions live under `firmware/src/`. Host and MCXN236 platform adapters use the same runtime and kernel code. The board application runs optimized and reference inference against golden input/output vectors and reports cycle and timing measurements through the platform console.

## Models and execution

| Model | Input | Main operations | Output |
| --- | --- | --- | --- |
| `mlp_demo` | 384 INT8 values | Fully connected layers, ReLU, softmax | 4 classes |
| `cnn_demo` | 128 × 3 INT8 sequence | 1D convolution, max pooling, fully connected, softmax | 4 classes |

Weights and activations use INT8 values; biases and accumulators use 32-bit integers. Quantization parameters and layer shapes are exported with each model. Intermediate storage is provided by the caller through a fixed-size arena; inference does not allocate memory dynamically.

## Repository structure

```text
firmware/
  src/                 Portable runtime, kernels, arena, and platform interfaces
  mcxn236/             FRDM-MCXN236 application and SDK project definition
models/
  fp32/                Trained models, dataset split, and normalization artifacts
  quant/               Quantized graphs and tensor archives
  export/              Generated C model data, metadata, and golden vectors
tools/                 Training, quantization, export, simulation, and debug scripts
tests/host/            C host tests for kernels and full-model inference
docs/                  Quantization, model metadata, and golden-vector contracts
scripts/check_all.sh   Host and model-pipeline checks
```

## Build and run host tests

Requirements: CMake 3.16 or newer and a C compiler. From the repository root:

```sh
cmake -S tests/host -B build/host
cmake --build build/host --parallel
ctest --test-dir build/host --output-on-failure
```

The host suite covers a basic toolchain check, kernel behavior, and full-model comparisons against the checked-in golden vectors.

## Reproduce the model pipeline

Python dependencies are listed in `requirements.txt` (PyTorch and NumPy). With those installed in your environment, run:

```sh
python tools/train.py
python tools/quantize.py
python tools/simulate.py
python tools/export_model.py
```

The scripts write generated artifacts beneath `models/`. The simulator evaluates quantized inference against the saved data split; the exporter produces the C model arrays and metadata consumed by firmware. See [`tools/README.md`](tools/README.md) and the contracts in [`docs/`](docs/) for implementation details.

## Build for FRDM-MCXN236

The target application is in `firmware/mcxn236/`. It uses the MCUXpresso SDK board support and requires an installed SDK checkout. Configure with the path to the SDK's `mcuxsdk` directory and its board project root, then build using the SDK CMake workflow. The project definition expects the SDK board port at `examples/_boards/frdmmcxn236/demo_apps/hello_world`. The application verifies both exported models against golden vectors and prints cycle and timing measurements over the configured console.

## Scope and limitations

This project demonstrates integer inference and validation using generated sensor-like data. The checked-in accuracy results apply only to that dataset. The firmware project provides a board verification application, but performance depends on building and running it on the target hardware. No measured speedup, RAM/Flash reduction, energy use, or field-data accuracy is claimed here. The runtime supports the included model operators and is not a general-purpose ONNX or TensorFlow Lite engine.
