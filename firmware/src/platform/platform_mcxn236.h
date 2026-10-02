#ifndef TINYML_PLATFORM_MCXN236_H
#define TINYML_PLATFORM_MCXN236_H
#include <stdint.h>
void hal_init(void);
uint64_t hal_get_cycle_count(void);
uint32_t hal_get_cpu_freq_hz(void);
void hal_print(const char *s);
void hal_print_int(const char *s, int64_t v);
#endif
