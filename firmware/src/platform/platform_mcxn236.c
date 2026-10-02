/* MCUXpresso SDK + board definitions are supplied by the board project. */
#include "platform_mcxn236.h"
#include "fsl_common.h"
#include "fsl_debug_console.h"

void hal_init(void) {
#if defined(DWT) && defined(CoreDebug_DEMCR_TRCENA_Msk) && defined(DWT_CTRL_CYCCNTENA_Msk)
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
#endif
}

uint64_t hal_get_cycle_count(void) {
#if defined(DWT) && defined(DWT_CTRL_CYCCNTENA_Msk)
    return (uint64_t)DWT->CYCCNT;
#else
    return 0U;
#endif
}

uint32_t hal_get_cpu_freq_hz(void) { return SystemCoreClock; }
void hal_print(const char *s) { PRINTF("%s", s); }

/* Avoid long-long printf specifiers: the SDK's compact console formatter
 * emits them literally. Convert the value to decimal and print as a string. */
void hal_print_int(const char *s, int64_t v) {
    char digits[20];
    size_t n = 0U;
    uint64_t magnitude;
    int negative = (v < 0);
    if (negative) {
        magnitude = (uint64_t)(-(v + 1)) + 1U;
    } else {
        magnitude = (uint64_t)v;
    }
    do {
        digits[n++] = (char)('0' + (magnitude % 10U));
        magnitude /= 10U;
    } while (magnitude != 0U);
    hal_print(s);
    if (negative) hal_print("-");
    while (n > 0U) {
        char digit[2] = { digits[--n], '\0' };
        hal_print(digit);
    }
    hal_print("\r\n");
}
