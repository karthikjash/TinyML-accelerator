cmake_minimum_required(VERSION 3.16)
project(tinyml_host_tests C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Global optimization flags — the benchmark fairness rule requires identical
# flags for naive and optimized code paths, set here, not per-file.
set(CMAKE_C_FLAGS "-Wall -Wextra -O3 -fsigned-char")

# Host backend of the platform HAL (firmware/src/platform/platform.h).
add_compile_definitions(PLATFORM_HOST)

# firmware headers live at repo root; this file is configured with -S tests/host,
# so resolve the root relative to this CMakeLists.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../firmware/src)

# Day 1: a hello test proving the host toolchain (GATE 0 requirement).
add_executable(hello_test hello_test.c)
add_test(NAME hello COMMAND hello_test)

enable_testing()
