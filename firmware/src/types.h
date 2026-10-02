/*
 * types.h — fixed-width integer contract for ALL kernels and runtime code.
 *
 * Rationale: plain `char` is signed on x86 and unsigned in some Arm ABI
 * configurations; int8_t/int32_t everywhere removes a whole class of
 * host-vs-target mismatches (guide, Debugging Map).
 */
#ifndef TINYML_TYPES_H
#define TINYML_TYPES_H

#include <stdint.h>
#include <stddef.h>

typedef int8_t   q7_t;    /* quantized int8 activation / weight */
typedef int32_t  q31_t;   /* accumulator / bias */
typedef int64_t  q63_t;   /* wide MAC accumulation before int32 clamp */

#endif /* TINYML_TYPES_H */
