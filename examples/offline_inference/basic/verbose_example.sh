export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION=1

CMD="python basic.py 2>&1 | tee basic.log"
echo $CMD
eval $CMD
