# Configurable TinyML Acceleration Framework for NXP MCXN236

A configurable, model-aware TinyML execution runtime targeting efficient INT8 neural-network inference on the **NXP FRDM-MCXN236** (Arm Cortex-M33).

---

## 1. Project Overview & Key Objectives

Microcontrollers face strict bounds on CPU, RAM, Flash, memory bandwidth, and power. This framework replaces conventional hard-coded implementations with a model-aware execution layer that configures optimized kernels based on offline model structure analysis.

### Primary Objectives
1. **Target:** NXP FRDM-MCXN236 (Arm Cortex-M33).
2. **Quantization:** INT8 / fixed-point inference (offline quantization pipeline).
3. **Core Kernels:** MAC, GEMM/FC, 1D Convolution, Activations.
4. **Supported Model Classes:** Multilayer Perceptron (MLP) & 1D Convolutional Neural Network (1D-CNN).
5. **Memory Strategy:** Dynamic-allocation-free static buffering & tensor reuse.
6. **Benchmarking:** Automated baseline vs. optimized comparison across Cycles, Latency, Peak RAM, Flash, Model Size, Accuracy, and Energy (optional).

---

## 2. Framework Workflow & Execution Pipeline

```text
+-------------------------------------------------------------------+
|                           OFFLINE SIDE                            |
| Model Training -> INT8 Quantization -> Model Analysis & Config    |
+---------------------------------+---------------------------------+
                                  | (Generated Code + Config Headers)
                                  v
+-------------------------------------------------------------------+
|                           MCXN236 SIDE                            |
| Model Interface -> Buffer Manager & Dispatcher -> Optimized Kernels|
|                       -> Inference -> Benchmarking                |
+-------------------------------------------------------------------+
```

### Core Architecture Layers
- **Model Analyzer (Offline):** Extracts layer shapes, channels, quantization scales, tensor lifespans, and selects required kernels.
- **Kernel Configuration System:** Pre-compiles execution paths to eliminate runtime setup overhead (*Configure offline, execute online*).
- **TinyML Runtime (Embedded):** Lightweight C/C++ engine handling static tensor buffers and zero-copy kernel invocation via a thin dispatcher.
- **CMSIS-NN Baseline:** Utilizes CMSIS-NN/reference kernels as performance baselines.

---

## 3. Supported Model Architectures

| Model Class | Computational Operations | Target Quantization & Types |
| :--- | :--- | :--- |
| **MLP** | $y = W \cdot x + b$ | Weights/Activations: **INT8**, Bias/Accumulator: **32-bit Integer** |
| **1D-CNN** | $y[t] = \sum x[t+k] \cdot w[k] + b$ | Weights/Activations: **INT8**, Sliding-window buffer reuse |

---

## 4. Quantization & Memory Model

- **Data Flow:** Float32 Model $\rightarrow$ INT8 Calibration/Quantization $\rightarrow$ Header Export (`model_data.c/.h`).
- **Memory Allocation:** Zero dynamic memory allocation (`malloc`) during inference. Static pre-allocated buffer layout with lifespan-based overlapping intermediate storage.

```text
+-------------------------------------------------------+
| Input Buffer | Intermediate Buffer 1 | Buffer 2 | Output|
+-------------------------------------------------------+
```

---

## 5. Benchmarking & Correctness Protocol

### Validation Strategy
1. **Correctness First:** Unit/layer output verified against reference floating-point/INT8 models within bounds before benchmarking.
2. **Fairness Guarantee:** Baseline and optimized implementations execute identical model graphs, precision, inputs, and compiler flags (`-O3`).

### Core Evaluation Metrics
- **Inference Latency & CPU Cycles:** Measured using MCU hardware cycle counter (DWT/SysTick).
- **Memory Footprint:** Peak RAM and total Flash usage.
- **Model Performance:** Quantized accuracy vs. FP32 baseline.
- **Speedup Ratio:** $\text{Speedup} = \frac{\text{Baseline Cycles}}{\text{Optimized Cycles}}$

---

## 6. Repository Layout

```text
tinyml-mcxn236/
├── firmware/
│   ├── src/
│   │   ├── main.c
│   │   ├── runtime/     # Dispatcher & runtime engine
│   │   ├── kernels/     # MAC, GEMM, FC, Conv1D
│   │   ├── memory/      # Buffer manager
│   │   └── benchmark/   # Latency & cycle measurements
├── models/              # MLP & 1D-CNN trained weights/metadata
├── tools/               # Offline Quantization, Analyzer & Header Generators
├── tests/               # Unit, Layer & Integration tests
└── third_party/cmsis/   # CMSIS-NN Baseline reference
```

---

## 7. Development Rules & Boundaries

### In Scope
- INT8 MLP and 1D-CNN inference on Arm Cortex-M33 (FRDM-MCXN236).
- Custom/CMSIS-NN optimized kernels (MAC, GEMM, Conv1D).
- Offline-configured static memory layout and execution paths.

### Strictly Out of Scope
- MCU-side model training, dynamic graph compilation, or general-purpose ONNX/TFLite engines.
- Large-scale vision/LLM architectures, ASIC/FPGA acceleration.
- Dynamic memory allocation (`malloc`) during inference.


## Toolchain & Practical Validation (Added from Proposal)

### Toolchain & Development Environment
- **PC-Side Training & Conversion:** Python-based ML tools (TensorFlow/PyTorch) supporting training and **Transfer Learning**.
- **Target IDE & SDK:** NXP **MCUXpresso** environment using Cortex-M33 SIMD/integer processing capabilities.

### Application Case Study
- Beyond synthetic benchmarks, the framework will be validated using an **embedded sensor-based application/case study** to demonstrate real-time performance in a practical operating scenario.

BENCHMARKS FOR EVALUATION.

Target Hardware: NXP FRDM-MCXN236 (Arm Cortex-M33 @ 150 MHz)
Optimization Flags: -O3 -mcpu=cortex-m33

Model: 1D-CNN (Keyword Spotting / Sensor Signal)
========================================================================================
Metric                   Naive C        CMSIS-NN Baseline   Your Framework   Improvement vs CMSIS
----------------------------------------------------------------------------------------
Inference Cycles         X,XXX,XXX      YYY,YYY             ZZZ,ZZZ          +XX.X% Speedup
Latency (ms)             XX.X ms        Y.YY ms             Z.ZZ ms          -XX.X% Latency
Peak RAM (KB)            XX.X KB        YY.Y KB             ZZ.Z KB          -XX.X% RAM
Flash Footprint (KB)     XX.X KB        YY.Y KB             ZZ.Z KB          -XX.X% Code Size
Top-1 Accuracy (%)       94.2%          94.1%               94.1%            0.0% Loss
========================================================================================
