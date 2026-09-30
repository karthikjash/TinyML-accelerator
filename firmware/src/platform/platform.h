/*
 * platform.h — the hardware abstraction layer (one header, two backends).
 *
 * All target-specific code in this project lives behind these calls.
 * Host build:   compile with -DPLATFORM_HOST
 * ESP32 build:  compiled inside the ESP-IDF project with -DPLATFORM_ESP32
 * MCXN236 port: add a third backend here — nothing else in the framework changes.
 */
#ifndef TINYML_PLATFORM_H
#define TINYML_PLATFORM_H

#include <stdint.h>
#include "types.h"

#if defined(PLATFORM_ESP32)
  #include "platform_esp32.h"
#elif defined(PLATFORM_HOST)
  #include "platform_host.h"
#else
  #error "Define PLATFORM_HOST or PLATFORM_ESP32"
#endif

#endif /* TINYML_PLATFORM_H */
