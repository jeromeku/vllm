INSTALL_CMD="VLLM_USE_PRECOMPILED=1 uv pip install --editable . -v"
echo "Installing vllm editable..." && eval ${INSTALL_CMD} 2>&1 | tee _vllm.build.log