/* Physical-board reference-versus-optimized TinyML benchmark demo. */
#include "board.h"
#include "app.h"
#include "platform/platform.h"
#include "runtime/models.h"
#include "../../models/export/cnn_demo/model_data.h"
#include "mcuxsdk_version.h"

#define DEMO_SAMPLE_COUNT 8U
#define DEMO_INPUT_COUNT 384U
#define DEMO_OUTPUT_COUNT 4U

extern const int8_t GOLDEN_MLP_DEMO_FULL_IN[DEMO_SAMPLE_COUNT * DEMO_INPUT_COUNT];
extern const int8_t GOLDEN_MLP_DEMO_FULL_OUT[DEMO_SAMPLE_COUNT * DEMO_OUTPUT_COUNT];
extern const int8_t GOLDEN_CNN_DEMO_FULL_IN[DEMO_SAMPLE_COUNT * DEMO_INPUT_COUNT];
extern const int8_t GOLDEN_CNN_DEMO_FULL_OUT[DEMO_SAMPLE_COUNT * DEMO_OUTPUT_COUNT];

static uint8_t arena[CNN_DEMO_ARENA_SIZE];

typedef struct {
    uint64_t reference_total;
    uint64_t optimized_total;
    size_t reference_golden_pass;
    size_t optimized_golden_pass;
    size_t reference_optimized_match;
    int passed;
} benchmark_summary_t;

static void print_uint64_inline(uint64_t value) {
    char digits[20];
    size_t count = 0U;
    do {
        digits[count++] = (char)('0' + (value % 10U));
        value /= 10U;
    } while (value != 0U);
    while (count > 0U) {
        char digit[2] = { digits[--count], '\0' };
        hal_print(digit);
    }
}

static void print_int64_inline(int64_t value) {
    uint64_t magnitude;
    if (value < 0) {
        hal_print("-");
        magnitude = (uint64_t)(-(value + 1)) + 1U;
    } else {
        magnitude = (uint64_t)value;
    }
    print_uint64_inline(magnitude);
}

static void print_int64_line(const char *label, int64_t value) {
    hal_print(label);
    print_int64_inline(value);
    hal_print("\r\n");
}

static void print_fraction(const char *label, size_t value, size_t total,
                           const char *pass_word, const char *fail_word) {
    hal_print(label);
    print_uint64_inline((uint64_t)value);
    hal_print("/");
    print_uint64_inline((uint64_t)total);
    hal_print(value == total ? pass_word : fail_word);
    hal_print("\r\n");
}

static void print_percentage(const char *label, int64_t hundredths) {
    uint64_t magnitude;
    hal_print(label);
    if (hundredths < 0) {
        hal_print("-");
        magnitude = (uint64_t)(-(hundredths + 1)) + 1U;
    } else {
        magnitude = (uint64_t)hundredths;
    }
    print_uint64_inline(magnitude / 100U);
    hal_print(".");
    if ((magnitude % 100U) < 10U) hal_print("0");
    print_uint64_inline(magnitude % 100U);
    hal_print("%\r\n");
}

static void print_average(const char *label, uint64_t total, size_t samples) {
    uint64_t whole = total / samples;
    uint64_t thousandths = ((total % samples) * 1000U) / samples;
    hal_print(label);
    print_uint64_inline(whole);
    hal_print(".");
    if (thousandths < 100U) hal_print("0");
    if (thousandths < 10U) hal_print("0");
    print_uint64_inline(thousandths);
    hal_print(" cycles/sample\r\n");
}

static void print_sample_header(size_t sample) {
    hal_print("\r\nSample ");
    print_uint64_inline((uint64_t)sample);
    hal_print("\r\n");
}

static int benchmark_model(const char *label, const tinyml_model_t *model,
                           const int8_t *inputs, const int8_t *golden,
                           benchmark_summary_t *summary) {
    uint32_t reference_cycles[DEMO_SAMPLE_COUNT];
    uint32_t optimized_cycles[DEMO_SAMPLE_COUNT];
    int8_t reference_output[DEMO_OUTPUT_COUNT];
    const int8_t *output;
    uint32_t begin, end;
    size_t sample, j;
    int dwt_valid = 1;
    int warm_reference_status, warm_optimized_status;
    int all_passed;
    int64_t cycles_saved;
    int64_t improvement_hundredths;

    summary->reference_total = 0U;
    summary->optimized_total = 0U;
    summary->reference_golden_pass = 0U;
    summary->optimized_golden_pass = 0U;
    summary->reference_optimized_match = 0U;
    summary->passed = 0;

    /* Warm both paths with the same input before collecting any timings. */
    warm_reference_status = tinyml_infer_reference(model, inputs, arena,
                                                    model->arena_n, &output);
    warm_optimized_status = tinyml_infer(model, inputs, arena,
                                         model->arena_n, &output);
    if (warm_reference_status != 0 || warm_optimized_status != 0) {
        hal_print(label);
        hal_print(" warm-up failed; benchmark not run.\r\n");
        return 0;
    }

    hal_print("\r\n------------------------------------------------\r\n");
    hal_print(label);
    hal_print(" - REFERENCE vs OPTIMIZED\r\n");
    hal_print("------------------------------------------------\r\n");
    hal_print("Samples: 8 (same inputs for both paths)\r\n");

    for (sample = 0U; sample < DEMO_SAMPLE_COUNT; ++sample) {
        int reference_status, optimized_status;
        int reference_pass = 0, optimized_pass = 0, outputs_match = 0;
        const int8_t *sample_input = inputs + sample * DEMO_INPUT_COUNT;
        const int8_t *sample_golden = golden + sample * DEMO_OUTPUT_COUNT;

        /* Timed region: inference call only. No printing or comparison here. */
        begin = (uint32_t)hal_get_cycle_count();
        reference_status = tinyml_infer_reference(model, sample_input, arena,
                                                   model->arena_n, &output);
        end = (uint32_t)hal_get_cycle_count();
        reference_cycles[sample] = (uint32_t)(end - begin);
        summary->reference_total += reference_cycles[sample];
        if (reference_cycles[sample] == 0U) dwt_valid = 0;
        if (reference_status == 0) {
            for (j = 0U; j < DEMO_OUTPUT_COUNT; ++j)
                reference_output[j] = output[j];
        }

        begin = (uint32_t)hal_get_cycle_count();
        optimized_status = tinyml_infer(model, sample_input, arena,
                                       model->arena_n, &output);
        end = (uint32_t)hal_get_cycle_count();
        optimized_cycles[sample] = (uint32_t)(end - begin);
        summary->optimized_total += optimized_cycles[sample];
        if (optimized_cycles[sample] == 0U) dwt_valid = 0;

        if (reference_status == 0) {
            reference_pass = 1;
            for (j = 0U; j < DEMO_OUTPUT_COUNT; ++j) {
                if (reference_output[j] != sample_golden[j]) {
                    reference_pass = 0;
                    break;
                }
            }
            if (reference_pass) ++summary->reference_golden_pass;
        }
        if (optimized_status == 0) {
            optimized_pass = 1;
            for (j = 0U; j < DEMO_OUTPUT_COUNT; ++j) {
                if (output[j] != sample_golden[j]) {
                    optimized_pass = 0;
                    break;
                }
            }
            if (optimized_pass) ++summary->optimized_golden_pass;
        }
        if (reference_status == 0 && optimized_status == 0) {
            outputs_match = 1;
            for (j = 0U; j < DEMO_OUTPUT_COUNT; ++j) {
                if (reference_output[j] != output[j]) {
                    outputs_match = 0;
                    break;
                }
            }
            if (outputs_match) ++summary->reference_optimized_match;
        }

        print_sample_header(sample);
        print_int64_line("  Reference cycles : ", (int64_t)reference_cycles[sample]);
        print_int64_line("  Optimized cycles : ", (int64_t)optimized_cycles[sample]);
        print_int64_line("  Cycles saved     : ",
                         (int64_t)reference_cycles[sample] -
                         (int64_t)optimized_cycles[sample]);
        hal_print(reference_pass ? "  Reference result : PASS (golden)\r\n" :
                                   "  Reference result : FAIL (golden)\r\n");
        hal_print(optimized_pass ? "  Optimized result : PASS (golden)\r\n" :
                                   "  Optimized result : FAIL (golden)\r\n");
        hal_print(outputs_match ? "  Reference == optimized : MATCH\r\n" :
                                 "  Reference == optimized : MISMATCH\r\n");
    }

    cycles_saved = (int64_t)summary->reference_total -
                   (int64_t)summary->optimized_total;
    all_passed = dwt_valid && summary->reference_total != 0U &&
                 summary->reference_golden_pass == DEMO_SAMPLE_COUNT &&
                 summary->optimized_golden_pass == DEMO_SAMPLE_COUNT &&
                 summary->reference_optimized_match == DEMO_SAMPLE_COUNT;

    hal_print("\r\n------------------------------------------------\r\n");
    hal_print(label);
    hal_print(" SUMMARY\r\n------------------------------------------------\r\n");
    print_int64_line("Reference total cycles : ", (int64_t)summary->reference_total);
    print_int64_line("Optimized total cycles : ", (int64_t)summary->optimized_total);
    print_int64_line("Cycles saved           : ", cycles_saved);
    print_average("Reference avg/sample   : ",
                  summary->reference_total, DEMO_SAMPLE_COUNT);
    print_average("Optimized avg/sample   : ",
                  summary->optimized_total, DEMO_SAMPLE_COUNT);
    if (summary->reference_total != 0U) {
        improvement_hundredths =
            (cycles_saved * INT64_C(10000)) / (int64_t)summary->reference_total;
        print_percentage("Performance improvement: ", improvement_hundredths);
    } else {
        hal_print("Performance improvement: unavailable\r\n");
        all_passed = 0;
    }
    print_fraction("Reference vs golden    : ", summary->reference_golden_pass,
                   DEMO_SAMPLE_COUNT, " PASS", " FAIL");
    print_fraction("Optimized vs golden    : ", summary->optimized_golden_pass,
                   DEMO_SAMPLE_COUNT, " PASS", " FAIL");
    print_fraction("Reference vs optimized : ", summary->reference_optimized_match,
                   DEMO_SAMPLE_COUNT, " MATCH", " MISMATCH");
    if (!dwt_valid) hal_print("DWT measurement       : FAIL (counter unavailable)\r\n");
    hal_print(label);
    hal_print(all_passed ? " benchmark result   : PASS\r\n" :
                           " benchmark result   : FAIL\r\n");
    summary->passed = all_passed;
    return all_passed;
}

static void print_model_summary(const char *label,
                                const benchmark_summary_t *summary) {
    hal_print(label);
    hal_print("  ");
    print_uint64_inline(summary->reference_total);
    hal_print("  ");
    print_uint64_inline(summary->optimized_total);
    hal_print("  ");
    print_int64_inline((int64_t)summary->reference_total -
                       (int64_t)summary->optimized_total);
    hal_print("\r\n");
}

int main(void) {
    benchmark_summary_t mlp_summary, cnn_summary;
    int mlp_ok, cnn_ok, final_ok;
    int64_t mlp_improvement, cnn_improvement;

    BOARD_InitHardware();
    hal_init();

    hal_print("================================================\r\n");
    hal_print(" TinyML Accelerator - MCXN236 Reference vs Optimized\r\n");
    hal_print("================================================\r\n");
    hal_print("Target : NXP FRDM-MCXN236 / Cortex-M33\r\n");
    hal_print("SDK    : 26.06.00 (runtime ");
    hal_print(MCUXSDK_VERSION_FULL_STR);
    hal_print(")\r\n");
    hal_print("UART   : MCU-Link VCOM, COM3 @ 115200\r\n");
    hal_print("Timing : physical DWT; inference calls only\r\n");
    hal_print("Warm-up: one untimed call per path and model\r\n");

    mlp_ok = benchmark_model("MLP", &tinyml_mlp_demo,
                             GOLDEN_MLP_DEMO_FULL_IN, GOLDEN_MLP_DEMO_FULL_OUT,
                             &mlp_summary);
    cnn_ok = benchmark_model("CNN", &tinyml_cnn_demo,
                             GOLDEN_CNN_DEMO_FULL_IN, GOLDEN_CNN_DEMO_FULL_OUT,
                             &cnn_summary);

    hal_print("\r\n================================================\r\n");
    hal_print("          TINYML ACCELERATOR RESULTS\r\n");
    hal_print("Target: NXP FRDM-MCXN236 / Cortex-M33\r\n");
    hal_print("------------------------------------------------\r\n");
    hal_print("MODEL  REFERENCE  OPTIMIZED  SAVED (cycles)\r\n");
    print_model_summary("MLP", &mlp_summary);
    print_model_summary("CNN", &cnn_summary);

    if (mlp_summary.reference_total != 0U) {
        mlp_improvement = (((int64_t)mlp_summary.reference_total -
                            (int64_t)mlp_summary.optimized_total) * INT64_C(10000)) /
                          (int64_t)mlp_summary.reference_total;
        print_percentage("MLP improvement : ", mlp_improvement);
    } else {
        hal_print("MLP improvement : unavailable\r\n");
    }
    if (cnn_summary.reference_total != 0U) {
        cnn_improvement = (((int64_t)cnn_summary.reference_total -
                            (int64_t)cnn_summary.optimized_total) * INT64_C(10000)) /
                          (int64_t)cnn_summary.reference_total;
        print_percentage("CNN improvement : ", cnn_improvement);
    } else {
        hal_print("CNN improvement : unavailable\r\n");
    }

    print_fraction("MLP correctness : ", mlp_summary.optimized_golden_pass,
                   DEMO_SAMPLE_COUNT, " PASS", " FAIL");
    print_fraction("CNN correctness : ", cnn_summary.optimized_golden_pass,
                   DEMO_SAMPLE_COUNT, " PASS", " FAIL");
    final_ok = mlp_ok && cnn_ok;
    hal_print(final_ok ? "Reference == Optimized : PASS\r\n" :
                         "Reference == Optimized : FAIL\r\n");
    hal_print(final_ok ? "Optimized == Golden    : PASS\r\n" :
                         "Optimized == Golden    : FAIL\r\n");
    hal_print("================================================\r\n");
    hal_print(final_ok ?
              "PHYSICAL TARGET BENCHMARK: PASS\r\nDAY 3 TARGET VERIFICATION: PASS\r\n" :
              "PHYSICAL TARGET BENCHMARK: FAIL\r\nDAY 3 TARGET VERIFICATION: FAIL\r\n");
    hal_print("================================================\r\n");
    for (;;) { }
}
