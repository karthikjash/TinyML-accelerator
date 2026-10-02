#ifndef TINYML_KERNELS_H
#define TINYML_KERNELS_H
#include <stddef.h>
#include <stdint.h>
void tinyml_requantize(const int32_t *acc, size_t n, double scale, int32_t zp, int8_t *out);
void tinyml_fc(const int8_t *x, const int8_t *w, const int32_t *bias, size_t ni, size_t no, int32_t zp, int32_t *acc);
void tinyml_conv1d(const int8_t *x, size_t t, size_t ci, const int8_t *w, const int32_t *bias, size_t co, size_t k, int same, int32_t zp, int32_t *acc);
void tinyml_relu(int8_t *x, size_t n, int32_t zp);
void tinyml_maxpool1d(const int8_t *x, size_t t, size_t c, size_t window, size_t stride, int8_t *y);
void tinyml_softmax_lut(const int8_t *x, size_t n, const double lut[256], double scale_out, int32_t zp_out, int8_t *y);
int8_t tinyml_requantize_one(int32_t acc, double scale, int32_t zp);
/* Original loop-order references retained for bit-exact comparison. */
void tinyml_fc_quant_reference(const int8_t *x, const int8_t *w, const int32_t *bias, size_t ni, size_t no, int32_t zp_in, double scale, int32_t zp_out, int relu, int8_t *y);
void tinyml_conv1d_quant_reference(const int8_t *x, size_t t, size_t ci, const int8_t *w, const int32_t *bias, size_t co, size_t k, int same, int32_t zp_in, double scale, int32_t zp_out, int relu, int8_t *y);
/* Cortex-M-friendly kernels preserve the reference arithmetic and layout. */
void tinyml_fc_quant_optimized(const int8_t *x, const int8_t *w, const int32_t *bias, size_t ni, size_t no, int32_t zp_in, double scale, int32_t zp_out, int relu, int8_t *y);
void tinyml_conv1d_quant_optimized(const int8_t *x, size_t t, size_t ci, const int8_t *w, const int32_t *bias, size_t co, size_t k, int same, int32_t zp_in, double scale, int32_t zp_out, int relu, int8_t *y);
void tinyml_fc_quant(const int8_t *x, const int8_t *w, const int32_t *bias, size_t ni, size_t no, int32_t zp_in, double scale, int32_t zp_out, int relu, int8_t *y);
void tinyml_conv1d_quant(const int8_t *x, size_t t, size_t ci, const int8_t *w, const int32_t *bias, size_t co, size_t k, int same, int32_t zp_in, double scale, int32_t zp_out, int relu, int8_t *y);
#endif
