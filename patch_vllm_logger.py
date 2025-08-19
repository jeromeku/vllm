import glob
import vllm.logger as vllm_log 
from pathlib import Path

if __name__ == "__main__":
    LOG_FILE_PAT = "%(filename)s"
    LOG_PATHNAME_PAT = "%(pathname)s"
    vllm_log_file = Path(vllm_log.__file__)
    print(f">>> Patching {vllm_log_file}")
    with open(vllm_log_file) as f:
        lines = f.readlines()
    # Check
    assert sum(1 for line in lines if LOG_FILE_PAT in line) == 1
    with open(vllm_log_file, 'w') as f:
        for line in lines:
            if LOG_FILE_PAT in line:
                print(f"Replacing {line}")
                line = line.replace(LOG_FILE_PAT, LOG_PATHNAME_PAT)
            f.write(line)
    print(f"<<< done patching {vllm_log_file}")
