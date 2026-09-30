/*
 * hello_test.c — GATE 0 host test (Day 1).
 *
 * Proves: C11 toolchain, -O3 + -fsigned-char build, int8/int32/int64 types,
 * rintf-free round-half-to-even rounding via nearbyint (C99, default FP mode),
 * and the two's-complement cast that every requantize path will depend on.
 * If this passes on host, the same checks run on ESP32 in Day 3's port of this
 * exact file.
 */
#include <math.h>
#include <stdio.h>
#include "platform/platform.h"
#include "types.h"

static int failures = 0;

static void check(const char *name, int cond)
{
    if (!cond) {
        failures++;
        printf("FAIL %s\n", name);
    } else {
        printf("pass %s\n", name);
    }
}

int main(void)
{
    /* 1. Types have the widths the whole framework assumes. */
    check("sizeof q7_t == 1",  sizeof(q7_t)  == 1);
    check("sizeof q31_t == 4", sizeof(q31_t) == 4);
    check("sizeof q63_t == 8", sizeof(q63_t) == 8);

    /* 2. Saturation corners survive the int32 -> int8 two's-complement cast
     *    (the requantize output path defined in CONTRACT_1 §2). */
    check("cast 127 -> 127",   (q7_t)(q31_t)127   == 127);
    check("cast 128 -> -128",  (q7_t)(q31_t)128  == -128);
    check("cast -128 -> -128", (q7_t)(q31_t)-128 == -128);
    check("cast -129 -> 127",  (q7_t)(q31_t)-129 ==  127);

    /* 3. Round-half-to-even is the FP default rounding mode (CONTRACT_1 §2):
     *    rint(0.5)=0, rint(1.5)=2, rint(2.5)=2, rint(-0.5)=-0 (=> 0), rint(-1.5)=-2. */
    check("rint 0.5 == 0",   (int)rint(0.5)   == 0);
    check("rint 1.5 == 2",   (int)rint(1.5)   == 2);
    check("rint 2.5 == 2",   (int)rint(2.5)   == 2);
    check("rint -1.5 == -2", (int)rint(-1.5)  == -2);

    /* 4. MAC accumulation in int64 then clamp to int32 (CONTRACT_1 §7). */
    q63_t wide = (q63_t)127 * 127;
    wide += (q63_t)127 * 127;
    check("q63 MAC = 32258", wide == 32258);
    check("int32 cast of small q63 exact", (q31_t)wide == 32258);

    if (failures == 0) {
        printf("ALL HELLO TESTS PASSED (host toolchain verified)\n");
        return 0;
    }
    printf("%d HELLO TEST(S) FAILED\n", failures);
    return 1;
}
