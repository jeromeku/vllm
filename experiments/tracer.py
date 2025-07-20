import datetime
import functools
import inspect
import json
import sys
from functools import partial
from types import FrameType
from typing import Any, Dict


# --------------------------------------------------------------------------- #
# 1.  SAFER  _make_jsonable()
# --------------------------------------------------------------------------- #
def _make_jsonable(obj: Any, param_name: str = "") -> Any:
    """
    Return *obj* if it's already JSON‑serialisable.
    Otherwise return a short sentinel string:
        "<non-jsonable {param_name}:{type(obj).__name__}>"
    This avoids calling user-defined __repr__ that might crash.
    """
    try:
        json.dumps(obj)          # Fast probe: will this serialise?
        return obj
    except (TypeError, OverflowError):
        type_name = type(obj).__name__
        label = f"{param_name}:{type_name}" if param_name else type_name
        return f"<non-jsonable {label}>"


# --------------------------------------------------------------------------- #
# 2.  TRACER
# --------------------------------------------------------------------------- #
def _trace_calls(log_path: str, root_dir: str,
                 frame: FrameType, event: str, arg: Any = None):
    if event not in ("call", "return"):
        return functools.partial(_trace_calls, log_path, root_dir)

    filename = frame.f_code.co_filename
    if not filename.startswith(root_dir):
        return functools.partial(_trace_calls, log_path, root_dir)

    lineno = frame.f_lineno
    func_name = frame.f_code.co_name

    caller = frame.f_back
    caller_path = caller.f_code.co_filename if caller else ""
    caller_lineno = caller.f_lineno if caller else 0

    entry: Dict[str, Any] = {
        "ts": datetime.datetime.now().isoformat(timespec="microseconds"),
        "event": event,
        "func": func_name,
        "location": f"{filename}:{lineno}",
        "caller": f"{caller_path}:{caller_lineno}",
    }

    if event == "call":
        arg_info = inspect.getargvalues(frame)

        # Positional & *args
        entry["args"] = [
            _make_jsonable(frame.f_locals[name], name) for name in arg_info.args
        ]

        # Keyword‑only & **kwargs (anything in locals not already listed)
        entry["kwargs"] = {
            name: _make_jsonable(value, name)
            for name, value in frame.f_locals.items()
            if name not in arg_info.args
        }

    else:  # "return"
        entry["return"] = _make_jsonable(arg, "return")

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    except NameError:
        # Globals may be torn down during interpreter shutdown.
        pass

    return functools.partial(_trace_calls, log_path, root_dir)


def enable_trace(log_path: str, root_dir: str):
    print(f"Tracing all function calls in {root_dir}, saving trace to {log_path}")
    sys.settrace(partial(_trace_calls, log_path, root_dir))