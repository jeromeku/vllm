#!/bin/bash

SCRIPT=./examples/offline_inference/basic/basic.py

# Check if the SCRIPT exists
if [ ! -f "$SCRIPT" ]; then
  echo "Error: $SCRIPT does not exist."
  exit 1
fi

# Parse out the file stem of the script
FILE_STEM=$(basename "$SCRIPT" .py)

# formats: flamegraph (default), raw, speedscope, chrometrace

CMD="py-spy record --subprocesses --native --format speedscope -o ${FILE_STEM}.spy.json -- python $SCRIPT"

echo $CMD
eval $CMD

