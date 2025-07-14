import sys
import threading
import torch.distributed as c10d
import torch
import torch.distributed as dist
from functools import wraps, partial

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    MultiThreadedTestCase,
)

from torch.testing._internal.distributed.multi_threaded_pg import (
    _install_threaded_pg,
    _uninstall_threaded_pg,
    ProcessLocalGroup,
)
def spawn_threads_and_init_comms(
    func=None, timeout=300, world_size=1
):
    """
    Wrapper to use with a test method
    """
    if func is None:
        return partial(
            spawn_threads_and_init_comms, timeout=timeout, world_size=world_size
        )


    def _run_test_method_with_multi_threads(world_size, callback):
        world = _install_threaded_pg()
        global_store = c10d.HashStore()

        def world_is_valid():
            return world == c10d.distributed_c10d._world

        def worker(rank, world_pg, store):
            c10d.init_process_group(
                backend="threaded", rank=rank, world_size=world_size, store=store
            )
            try:
                callback()
            except BaseException as ex:
                # Exceptions are handled in MultiThreadedTestCase
                MultiThreadedTestCase.exception_queue.put((rank, sys.exc_info()))
                ProcessLocalGroup.exception_handle(ex)  # trigger _terminate event and awaken worker threads
            finally:
                if world_is_valid():
                    c10d.destroy_process_group()

        threads = []
        for rank in range(world_size):
            t = threading.Thread(target=worker, args=(rank, world, global_store))
            t.start()
            threads.append(t)

        return threads


    @wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: get test name from kwargs
        torch._C._distributed_c10d._set_thread_isolation_mode(True)
        try:
            threads = _run_test_method_with_multi_threads(world_size, lambda: func(*args, **kwargs))
            # join and error handling
            MultiThreadedTestCase._join_threads(threads, func)
        finally:
            torch._C._distributed_c10d._set_thread_isolation_mode(False)

    return wrapper


def dist_print(msg):
    import time
    rank = dist.get_rank()
    time.sleep(rank)
    print(f"{rank=}: {msg}", flush=True)

@spawn_threads_and_init_comms(world_size=2)
def test_all_to_all_single_tensor():
    dtype = torch.float
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_size = world_size + 1
    send = torch.arange(rank, rank + world_size, dtype=dtype)
    dist_print(f"{send=}")
    
    # rank 0: 0, 1
    # rank 1: 1, 2
    # rank 0 send 1 => rank 1
    # rank 1 doesn't send anything
    # rank 0 out = [0], rank1 out = [1, 1, 2]

    if rank == 0:
        input_splits = [1, 1]
        output_splits = [1, 0]
    else:
        input_splits = [0, 2]
        output_splits = [1, 2]
    out = torch.empty(sum(output_splits), dtype=dtype)
    # sizes = torch.tensor([1, ])

    #out = torch.zeros(world_size, 2, dtype=send.dtype)
    dist.all_to_all_single(out, send, output_split_sizes=output_splits, input_split_sizes=input_splits)
    dist_print(f"{out.tolist()=}")

    #assert out.tolist() == list(zip(range(world_size), range(world_size)))

test_all_to_all_single_tensor()