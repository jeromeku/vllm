#!/bin/bash

rm -rf build && cmake -Bbuild -S. -DCMAKE_VERBOSE_MAKEFILE=1 -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -GNinja 2>&1 | tee cmake.build.log 