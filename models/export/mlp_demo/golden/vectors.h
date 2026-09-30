/* vectors.h — AUTO-GENERATED golden vectors for mlp_demo. DO NOT EDIT. Contract 3. */
#ifndef GOLDEN_MLP_DEMO_H
#define GOLDEN_MLP_DEMO_H

#include <stdint.h>

typedef struct { const char *name; int in_len; const int8_t *in; int out_len; const int8_t *out; } golden_vec_t;

#define GOLDEN_MLP_DEMO_N_LAYERS 4
extern const golden_vec_t GOLDEN_MLP_DEMO_LAYERS[GOLDEN_MLP_DEMO_N_LAYERS];

#define GOLDEN_MLP_DEMO_N_SAMPLES 8
#define GOLDEN_MLP_DEMO_IN_LEN 384
#define GOLDEN_MLP_DEMO_OUT_LEN 4
extern const int8_t GOLDEN_MLP_DEMO_FULL_IN[GOLDEN_MLP_DEMO_N_SAMPLES * GOLDEN_MLP_DEMO_IN_LEN];
extern const int8_t GOLDEN_MLP_DEMO_FULL_OUT[GOLDEN_MLP_DEMO_N_SAMPLES * GOLDEN_MLP_DEMO_OUT_LEN];

#endif /* GOLDEN_MLP_DEMO_H */
