#!/bin/bash

set -euo pipefail

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_CONFIG_PATH="vllm_log_config.json"

DT=$(date +"%Y%m%d%H%M")

LOG_DIR=logs
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${DT}.log

# export VLLM_LOGGING_STREAM="${LOGDIR}/${DT}.log"
export VLLM_DEBUG_DUMP_PATH="vllm_compile/${DT}"
mkdir -p ${VLLM_DEBUG_DUMP_PATH}

#export VLLM_PATTERN_MATCH_DEBUG=1
TORCH_LOGS=""
CMD="python test_functionalization.py" 

if [[ -n ${TORCH_LOGS} ]]; then
    RUN_CMD="TORCH_LOGS=${TORCH_LOGS} ${CMD}"
else
    RUN_CMD=$CMD
fi

echo ${RUN_CMD}
eval ${RUN_CMD} 2>&1 | tee ${LOG_PATH}