import inspect
from contextlib import contextmanager

import viztracer

from vllm.logger import logger

_VLLM_ROOT = None


def _get_vllm_root():
    global _VLLM_ROOT
    if _VLLM_ROOT is not None:
        return

    import os

    import vllm

    _VLLM_ROOT = os.path.dirname(os.path.abspath(vllm.__file__))
    logger.info(f"Setting VLLM_ROOT: {_VLLM_ROOT}")


DEFAULT_INCLUDES = [_get_vllm_root()]

DEFAULT_TRACER_SETTINGS = dict(
    include_files=DEFAULT_INCLUDES,
    ignore_frozen=True,
    ignore_c_function=True,
    log_func_args=True,
    log_func_retval=True,
    dump_raw=False,
    pid_suffix=True,
)


def get_tracer(
    include_files=DEFAULT_INCLUDES,
    ignore_frozen=True,
    ignore_c_function=True,
    log_func_args=True,
    log_func_retval=True,
    dump_raw=False,
    pid_suffix=True,
    **kwargs,
):
    frame = inspect.currentframe()
    args_info = inspect.getargvalues(frame)
    all_kwargs = args_info.locals.get(args_info.keywords)
    print(f"all kwarg: {all_kwargs}")

    tracer = viztracer.get_tracer()
    if tracer is None:
        tracer = viztracer.VizTracer(
            include_files=include_files,
            ignore_frozen=ignore_frozen,
            ignore_c_function=ignore_c_function,
            log_func_args=log_func_args,
            log_func_retval=log_func_retval,
            dump_raw=dump_raw,
            pid_suffix=pid_suffix,
            **kwargs,
        )
    else:
        tracer.include_files = include_files
        tracer.ignore_frozen = ignore_frozen
        tracer.ignore_c_function = ignore_c_function
        tracer.log_func_args = log_func_args
        tracer.log_func_retval = log_func_retval
        tracer.dump_raw = dump_raw
        tracer.pid_suffix = pid_suffix
        for k, v in kwargs.items():
            setattr(tracer, k, v)

    return tracer


@contextmanager
def vllm_debug_context(
    disable_multiprocessing=True,
    mp_spawn_method="spawn",
    log_level="DEBUG",
    **kwargs,
):
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = (
        "0" if disable_multiprocessing else "1"
    )
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = mp_spawn_method
    os.environ["VLLM_LOGGING_LEVEL"] = log_level
    for k, v in kwargs.items():
        os.environ[k] = str(v)

    yield
