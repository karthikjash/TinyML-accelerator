/*
 * platform_host.h — host backend of the platform HAL.
 * Fake cycle counter (host timing is not meaningful for MCU benchmarks);
 * real cycle counting happens on ESP32 (Day 3) and MCXN236 (port backlog).
 */
#ifndef TINYML_PLATFORM_HOST_H
#define TINYML_PLATFORM_HOST_H

#include <stdint.h>
#include <stdio.h>

static inline void     hal_init(void)                       { }
static inline uint64_t hal_get_cycle_count(void)            { return 0; }
static inline uint32_t hal_get_cpu_freq_hz(void)            { return 0; }
static inline void     hal_print(const char *s)             { fputs(s, stdout); }
static inline void     hal_print_int(const char *s, int64_t v)
{
    printf("%s%lld\n", s, (long long)v);
}

#endif /* TINYML_PLATFORM_HOST_H */
