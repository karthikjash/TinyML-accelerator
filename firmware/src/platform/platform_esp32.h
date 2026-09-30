/*
 * platform_esp32.h — ESP32 backend of the platform HAL.
 * Compiled only inside the ESP-IDF project (Day 3, on the user's machine).
 * Uses the ESP32 cycle counter for real benchmarking.
 */
#ifndef TINYML_PLATFORM_ESP32_H
#define TINYML_PLATFORM_ESP32_H

#include <stdint.h>
#include <stdio.h>
#include "esp_cpu.h"

static inline void     hal_init(void)            { }
static inline uint64_t hal_get_cycle_count(void) { return (uint64_t)esp_cpu_get_cycle_count(); }
static inline uint32_t hal_get_cpu_freq_hz(void) { return 240000000u; }
static inline void     hal_print(const char *s)  { printf("%s", s); }
static inline void     hal_print_int(const char *s, int64_t v) { printf("%s%lld\n", s, (long long)v); }

#endif /* TINYML_PLATFORM_ESP32_H */
