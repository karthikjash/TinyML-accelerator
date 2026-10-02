# DAY 3 REPORT — MCXN236 Runtime, Kernels & Hardware Validation ✅

**Date:** 2026-10-03 · **Repo:** karthikjash/TinyML-accelerator (local checkout: C:\Users\naira\Downloads\TinyML-accelerator-main\TinyML-accelerator-main)
**Sprint:** 4-day TinyML framework build · **Day 3 target:** NXP FRDM-MCXN236 / Cortex-M33
**Gates today:** host reference/optimized/golden checks ✅ · MCXN236 build/flash/UART ✅ · optimized target verification and DWT comparison ✅

---

## 1. What was done (in order)

### Stage 4a — Reference and optimized kernels

Audited the C runtime and kernel layouts. The fused Fully Connected (FC) kernel reads weights in [input][output] order; the prior output-at-a-time loop used a stride through adjacent outputs. Conv1D uses [input_channel][output_channel][kernel] weights and had repeated flattened indexing and padding checks in the MAC loops.

The original fused implementations remain available:

- tinyml_fc_quant_reference
- tinyml_conv1d_quant_reference

The normal runtime path, tinyml_infer, calls the optimized kernels. tinyml_infer_reference is retained for correctness comparisons.

**Fully Connected optimization**

- Processes four adjacent outputs as a tile, following the contiguous output dimension of the existing [input][output] layout.
- Reuses each centered input across the tile and reduces repeated index calculation.
- Accumulates inputs in the same order as the reference for each output.
- Uses native 32-bit accumulation only when a conservative bound proves that every partial sum and bias addition fits in int32. Otherwise it uses 64-bit accumulation and the same int32 clamp as the reference.

**Conv1D optimization**

- Uses pointer traversal for input rows and weights instead of recalculating full tensor offsets for every MAC.
- Preserves the existing [input_channel][output_channel][kernel] layout and reduction order.
- Skips out-of-range padded positions because padding equals the input zero point and therefore contributes exactly zero after zero-point subtraction.

The shared round-half-to-even requantization, saturation, zero points, ReLU, softmax LUT, scales, tensor shapes, and tensor layouts are unchanged. No assembly was added, and no golden vectors were edited.

### Stage 4b — Official MCXN236 firmware integration

The official FRDM-MCXN236 hello_world example was used first as the known-good board foundation. It was built, flashed, debugged, and its output was checked over MCU-Link VCOM before the TinyML runtime was integrated. The final project uses the official SDK board port for startup, board and system initialization, linker setup, and debug console.

The TinyML C runtime and Day 1/Day 2 model artifacts were preserved while adding the MCXN236 firmware application. The final optimized firmware linked at:

~~~text
firmware/mcxn236/build/debug/tinyml_mcxn236.elf
firmware/mcxn236/build/debug/tinyml_mcxn236.bin
~~~

Final build memory report:

~~~text
m_text:       98,968 bytes (9.45%)
m_data:        6,464 bytes (2.82%)
m_interrupts:    688 bytes (67.19%)
~~~

### Stage 4c — Host verification

The host suite checks individual optimized FC and Conv1D outputs against their reference versions, including output tails, same and valid padding, and the wide-accumulator fallback. The full-model test runs both runtime paths on every checked-in input and requires:

~~~text
optimized == reference == golden
~~~

### Stage 4d — Physical target verification and measurement

The optimized ELF was programmed through MCU-Link / LinkServer to the physical FRDM-MCXN236. The final application ran over the Cortex-M33 and emitted verification results over MCU-Link VCOM on COM3 at 115200 baud.

For each sample, the existing DWT cycle counter is read immediately around tinyml_infer; the eight inference deltas are summed and printed. Counts below are actual target readings for the full 8-sample suites.

---

## 2. Gate evidence (actual output)

### Host build and tests

~~~text
100% tests passed, 0 tests failed out of 3
Tests: hello, kernels, full_model
ALL KERNEL TESTS PASSED
mlp_demo full-model vectors: PASS (8 samples, optimized/reference/golden exact)
cnn_demo full-model vectors: PASS (8 samples, optimized/reference/golden exact)
~~~

### Final target UART capture

Captured from COM3 at 115200 baud:

~~~text
TinyML MCXN236 Day 3
SDK: 2026.06.00

MLP target verification
sample PASS: 0
sample PASS: 1
sample PASS: 2
sample PASS: 3
sample PASS: 4
sample PASS: 5
sample PASS: 6
sample PASS: 7
exact samples: 8
inference cycles (all samples): 24208930
MLP golden test: PASS

CNN target verification
sample PASS: 0
sample PASS: 1
sample PASS: 2
sample PASS: 3
sample PASS: 4
sample PASS: 5
sample PASS: 6
sample PASS: 7
exact samples: 8
inference cycles (all samples): 113813665
CNN golden test: PASS
DAY3 TARGET VERIFICATION: PASS
~~~

### Final results

| Verification | Result |
|---|---|
| Host CTest | 3/3 PASS |
| MLP host vectors | 8/8 exact |
| CNN host vectors | 8/8 exact |
| Reference vs optimized full-model outputs | Bit-exact |
| MCXN236 firmware build | PASS |
| Physical flash | PASS |
| UART execution | PASS |
| MLP target vectors | 8/8 exact |
| CNN target vectors | 8/8 exact |

| Model / suite | Reference cycles | Optimized cycles | Cycles saved | Measured improvement |
|---|---:|---:|---:|---:|
| MLP, 8 samples | 30,466,706 | 24,208,930 | 6,257,776 | 20.54% |
| CNN, 8 samples | 140,381,540 | 113,813,665 | 26,567,875 | 18.93% |

Improvement is calculated as (reference - optimized) / reference. The reference values are the previously measured physical-target DWT totals for the same golden suites. These values are suite totals, not per-sample values; no production-build performance claim is made.

---

## 3. Key decisions & why

| Decision | Why |
|---|---|
| Use the official MCUXpresso FRDM-MCXN236 board project as the hardware foundation | It supplies the verified device startup, linker, clock/system setup, board initialization, and debug console. |
| Retain reference FC and Conv1D kernels and reference inference | Optimization needs a trusted, runnable baseline for bit-exact comparison. |
| Keep the existing tensor and quantization contracts | A kernel optimization must not change the meaning of model data or golden outputs. |
| Tile FC across four output channels | The stored output channels are contiguous; a tile reuses each input and reduces repeated indexing while retaining each output's accumulation order. |
| Use a bounded int32 FC path with a wide fallback | Native-width operations are used only when the bias and all partial sums are proven safe; unsupported ranges keep the reference-compatible 64-bit path and int32 clamp. |
| Traverse Conv1D by pointers and skip zero-contribution padding | This avoids repeated offset arithmetic while preserving layout and reduction order. |
| Keep portable C; add no assembly | These initial Cortex-M33-friendly changes stay maintainable and keep the reference easy to compare. |
| Measure on the physical target | Host timing is not a substitute for the target DWT counter. |
| Leave golden vectors unchanged | Vectors remain an independent correctness oracle rather than being altered to fit an implementation. |

---

## 4. Versions used (and why)

| Tool | Version | Purpose |
|---|---|---|
| MCUXpresso SDK | 26.06.00 (runtime identifies itself as 2026.06.00) | Official FRDM-MCXN236 board support and firmware foundation |
| Arm GNU Toolchain | 14.2.1 | Cortex-M33 firmware compilation and linking |
| MCU-Link / LinkServer | LinkServer 26.6.137 | Physical target programming, debug connection, and run |
| MinGW GCC | 6.3.0 | Windows host C tests (C:\MinGW\bin\gcc.exe) |
| CMake | 3.30.0 | Host and MCUXpresso project builds |
| Ninja | 1.13.2 | Configured MCXN236 build generator |
| MinGW Make | CMake-selected MinGW Makefiles generator | Host test build |

Versions above are from the recorded build environment and its generated CMake configuration. No unrecorded board or compiler version is inferred.

---

## 5. Issues hit today & how they were fixed

1. **MCXN236 startup/linker/board project was initially unavailable to the automated integration.** The user supplied the official MCUXpresso SDK; the official FRDM-MCXN236 hello_world example was imported and verified as a physical-board baseline. TinyML was then integrated using that SDK board port rather than inventing startup or linker files.
2. **The initial host configure selected NMake although nmake.exe was absent.** The host tests were configured with the available MinGW compiler and the MinGW Makefiles generator.
3. **An existing VS Code debug session owned the MCU-Link core.** LinkServer could detect the probe but could not flash while the core was claimed. The session was stopped with Shift+F5; the probe was then released and flashing succeeded.
4. **The SDK compact console did not render %lld correctly.** UART initially showed the literal format suffix instead of the number. The target HAL now formats signed integers as decimal text before writing them, allowing actual DWT totals to be captured.
5. **The first simple tiled FC version regressed the MLP timing.** A native 32-bit accumulation fast path was added only with a conservative overflow bound and a 64-bit fallback. Host equality checks still passed, and the final on-target measurement improved over the reference. The report records the final measurement, not the intermediate iteration.

---

## 6. Repo state after Day 3

The project now contains the following relevant areas (generated build directories and model binaries omitted):

~~~text
.
├── DAY1_REPORT.md
├── DAY2_REPORT.md
├── DAY3_REPORT.md
├── README.md
├── TinyML_4Day_Guide.html
├── TinyML_4Day_Guide.pdf
├── docs/
│   ├── CONTRACT_1_REQUANTIZE_SPEC.md
│   ├── CONTRACT_2_MODEL_JSON_SCHEMA.md
│   └── CONTRACT_3_GOLDEN_VECTORS.md
├── firmware/
│   ├── mcxn236/
│   │   ├── CMakeLists.txt
│   │   ├── main.c
│   │   ├── model_data_cnn.c
│   │   ├── model_data_mlp.c
│   │   └── prj.conf
│   └── src/
│       ├── kernels/ (kernels.c, kernels.h)
│       ├── memory/ (arena.c, arena.h)
│       ├── platform/ (HAL headers and host/MCXN236 implementations)
│       ├── runtime/ (models.c/.h, runtime.c/.h)
│       └── types.h
├── models/
│   ├── fp32/
│   ├── quant/
│   └── export/
│       ├── mlp_demo/ (model data, metadata, manifest, golden vectors)
│       └── cnn_demo/ (model data, metadata, manifest, golden vectors)
├── scripts/
│   └── check_all.sh
├── tests/
│   └── host/ (CMakeLists.txt, hello_test.c, kernel_test.c, full_model_test.c)
└── tools/
    ├── tinyml/ (contract_math.py)
    ├── train.py
    ├── quantize.py
    ├── simulate.py
    ├── export_model.py
    ├── debug_layer_compare.py
    └── README.md
~~~

Day 3 kernel/runtime and host-test changes are in firmware/src/kernels/, firmware/src/runtime/, and tests/host/. The MCXN236 project is in firmware/mcxn236/. Day 1/Day 2 model exports and checked-in golden vectors remain unchanged.

**Documentation note:** the root README.md still contains pre-integration text saying the MCXN236 SDK project and flash workflow are not configured. This report reflects the later verified state; the README was left untouched in this documentation-only request.

---

## 7. How Day 3 works

~~~text
FP32 model and quantized export
             ↓
model_data + checked-in golden vectors
             ↓
TinyML C runtime
       ↙             ↘
reference kernels   optimized kernels
       ↘             ↙
     bit-exact comparison on host
             ↓
MCXN236 firmware → Cortex-M33 inference
             ↓
DWT cycle counter around tinyml_infer
             ↓
UART verification on MCU-Link VCOM
             ↓
compare each output with golden vectors
~~~

- tinyml_infer_reference selects the preserved FC and Conv1D reference kernels.
- Normal tinyml_infer selects the optimized kernels.
- Host tests compare the two paths and require both to equal each golden vector before target validation.
- The flashed application repeats the golden-vector checks on the physical MCXN236 and reports the aggregate DWT cycle total for each 8-sample suite.

### Reproduce the host checks

From the repository root on Windows with the recorded MinGW/CMake tools:

~~~powershell
cmake -S tests/host -B build/host-mingw -G "MinGW Makefiles" -DCMAKE_C_COMPILER=C:/MinGW/bin/gcc.exe
cmake --build build/host-mingw --parallel
ctest --test-dir build/host-mingw --output-on-failure
~~~

### Rebuild and run the configured MCXN236 project

The checked-in source uses the official SDK project configuration; the local build/debug directory is configured for the MCUXpresso SDK installation on this machine.

~~~powershell
cmake --build firmware/mcxn236/build/debug --parallel
LinkServer.exe flash --probe 04FF0ZKGMAFB4 MCXN236:FRDM-MCXN236 load firmware/mcxn236/build/debug/tinyml_mcxn236.elf
LinkServer.exe run --probe 04FF0ZKGMAFB4 --mode serial:COM3:115200 --pass-mark "DAY3 TARGET VERIFICATION: PASS" --fail-mark "DAY3 TARGET VERIFICATION: FAIL" --exit-timeout 60 MCXN236:FRDM-MCXN236 firmware/mcxn236/build/debug/tinyml_mcxn236.elf
~~~

The programming/run commands require the MCU-Link probe to be connected and COM3 to be free. This run command reloads the ELF and captures UART while checking the aggregate pass/fail marker.

---

## 8. Next process — Day 4 checklist

Day 4 has **not** been started. The original guide is ESP32-oriented, while this project was explicitly completed on MCXN236; continue from the verified MCXN236 implementation and document the actual target/toolchain rather than switching platforms.

The guide's remaining Stage 5 objectives are the benchmark deliverable and demo/reproducibility work:

1. Fill a results table for both models with cycles/inference, latency, peak RAM, flash, INT8 accuracy, and FP32 accuracy. Clearly state board, frequency, SDK, compiler, and build flags actually used.
2. For a future benchmark campaign, follow and record the guide's warm-up/median protocol and fairness conditions; do not compare builds with differing flags or inputs.
3. Document how to run the demo and how to interpret its results in the README.
4. Add/verify a reproducible one-command path from clean checkout through export, host tests, target build, and demo where supported.
5. Verify deterministic outputs and the no-dynamic-allocation requirement against the actual firmware sources.
6. Complete and review the final report/demo deliverables before any release tag.

Do not reinterpret the Day 3 8-sample DWT suite totals as Day 4's repeated-run median benchmark. Preserve today's exact correctness and cycle results as the verified baseline.

### Definition of Done — Day 3

| Requirement | Status |
|---|---|
| Existing Day 1/Day 2 artifacts preserved | ✅ |
| Official MCXN236 SDK integrated | ✅ |
| MCXN236 firmware builds | ✅ |
| Firmware flashes to physical board | ✅ |
| UART execution verified | ✅ COM3, 115200 baud |
| Reference kernels retained | ✅ |
| Optimized FC implemented | ✅ |
| Optimized Conv1D implemented | ✅ |
| Host CTest | ✅ 3/3 |
| MLP host golden vectors | ✅ 8/8 |
| CNN host golden vectors | ✅ 8/8 |
| MLP target golden vectors | ✅ 8/8 |
| CNN target golden vectors | ✅ 8/8 |
| Optimized/reference bit-exact | ✅ |
| MLP DWT measurement | ✅ 24,208,930 cycles |
| CNN DWT measurement | ✅ 113,813,665 cycles |
| Day 4 started | ❌ |

# DAY 3 COMPLETE ✅
