#!/bin/bash

set -euo pipefail
rm -rf build
CMAKE_EXTRA_FLAGS="-DCMAKE_VERBOSE_MAKEFILE=1 -DCMAKE_EXPORT_COMPILE_COMMANDS=1"
GENERATOR=Ninja
CMD="cmake --regenerate-during-build -Bbuild -S. -G${GENERATOR} ${CMAKE_EXTRA_FLAGS} --debug-output"
echo $CMD
eval $CMD 2>&1 | tee _config.log
