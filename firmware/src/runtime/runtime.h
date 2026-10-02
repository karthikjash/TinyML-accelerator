#ifndef TINYML_RUNTIME_H
#define TINYML_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
typedef enum { TINYML_FC, TINYML_CONV1D, TINYML_MAXPOOL1D, TINYML_FLATTEN, TINYML_SOFTMAX } tinyml_op_t;
typedef struct { tinyml_op_t op; size_t in_n,out_n,in_t,in_c,out_t,out_c,kernel,stride; const int8_t *weights; const int32_t *bias; const double *lut,*scale_in,*scale_w,*scale_out; const int *zp_in,*zp_out; int same,relu; } tinyml_layer_t;
typedef struct { const char *name; const tinyml_layer_t *layers; size_t n_layers,input_n,arena_n,even_n,odd_n,output_n; } tinyml_model_t;
int tinyml_infer(const tinyml_model_t *model,const int8_t *input,void *arena,size_t arena_bytes,const int8_t **output);
int tinyml_infer_reference(const tinyml_model_t *model,const int8_t *input,void *arena,size_t arena_bytes,const int8_t **output);
#endif
