#!/usr/bin/env bash
# check_all.sh — run every gate that is currently implemented.
# Day 1: host hello test (GATE 0). Later days append: golden tests, benchmarks.
set -e
cd "$(dirname "$0")/.."

fail=0

echo "== [1/1] Host toolchain tests (GATE 0) =="
if [ ! -d build/host ]; then
  mkdir -p build/host
  cmake -S tests/host -B build/host > /dev/null
fi
cmake --build build/host > /dev/null
if ./build/host/hello_test; then
  echo "GATE 0 (host part): PASS"
else
  echo "GATE 0 (host part): FAIL"
  fail=1
fi

exit $fail
