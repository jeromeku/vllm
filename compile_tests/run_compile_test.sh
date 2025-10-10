#!/bin/bash

set -euo pipefail

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=DEBUG

DT=$(date +"%Y%m%d%H%M")

LOG_DIR=logs
mkdir -p ${LOG_DIR}
LOG_PATH=${LOG_DIR}/${DT}.log

# export VLLM_LOGGING_STREAM="${LOGDIR}/${DT}.log"
export VLLM_DEBUG_DUMP_PATH="vllm_compile/${DT}"

# Generate dynamic logging config with timestamp-based log filename
VLLM_LOG_CONFIG="vllm_log_config.${DT}.json"
VLLM_LOG_FILENAME="${LOG_DIR}/${DT}.vllm.debug.log"

cat > ${VLLM_LOG_CONFIG} <<EOF
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "vllm": {
            "class": "vllm.logging_utils.NewLineFormatter",
            "datefmt": "%m-%d %H:%M:%S",
            "format": "%(levelname)s %(asctime)s [%(pathname)s:%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "formatter": "vllm",
            "level": "DEBUG",
            "filename": "${VLLM_LOG_FILENAME}",
            "mode": "w"
        }
    },
    "loggers": {
        "vllm": {
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": false
        }
    }
}
EOF

export VLLM_LOGGING_CONFIG_PATH="${VLLM_LOG_CONFIG}"

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