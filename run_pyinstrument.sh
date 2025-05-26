#!/bin/bash

SCRIPT=./examples/offline_inference/basic/basic.py

# Check if the SCRIPT exists
if [ ! -f "$SCRIPT" ]; then
  echo "Error: $SCRIPT does not exist."
  exit 1
fi

# Parse out the file stem of the script
FILE_STEM=$(basename "$SCRIPT" .py)

CMD="pyinstrument -o ${FILE_STEM}.profile.txt --renderer text --timeline --show-regex=vllm --color --unicode $SCRIPT"

echo $CMD


