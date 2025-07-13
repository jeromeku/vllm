import operator
import os
import sys
import threading
from functools import reduce
from unittest import skip, SkipTest

import torch
import torch.autograd
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    MultiThreadedTestCase,
    skip_if_lt_x_gpu,
    spawn_threads_and_init_comms,
)
from torch.testing._internal.common_utils import IS_SANDCASTLE, run_tests, TestCase

def dist_print(msg):
    import time
    rank = dist.get_rank()
    time.sleep(rank)
    print(f"{rank=}: {msg}", flush=True)
@spawn_threads_and_init_comms(world_size=2)
def test_all_to_all_single_tensor(_):
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

test_all_to_all_single_tensor(None)