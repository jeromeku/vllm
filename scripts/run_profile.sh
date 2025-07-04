#!/bin/bash

set -euo pipefail
source ~/.bash_profile

THIS_DIR="$(dirname "$(realpath "$0")")"
VLLM_TORCH_PROFILER_DIR=${THIS_DIR}/profiles
MODEL_ID="Qwen/Qwen3-4B"
SCRIPT="${THIS_DIR}/profile_qwen3.py"
LOG_LEVEL="DEBUG"
NUM_TOKENS=5
NUM_TRIALS=1
NUM_PROMPTS=128
CONTEXT_LENGTH=512
export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR}"
export VLLM_LOGGING_LEVEL="${LOG_LEVEL}"
LOG_FILE=${THIS_DIR}/profiles/prof.`current_time`.log
echo "Writing to ${LOG_FILE}"

CMD="python ${SCRIPT} --model ${MODEL_ID} --n_trials ${NUM_TRIALS} --gen_tokens ${NUM_TOKENS} --n_prompts ${NUM_PROMPTS} --context_length ${CONTEXT_LENGTH}"
echo $CMD
eval $CMD 2>&1 | tee ${LOG_FILE}