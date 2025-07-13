# ruff: noqa
from collections import Counter
import functools
import weakref
from typing import Dict

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
import torchvision.models as models

class TensorMemoryProfilerMode(TorchDispatchMode):
    def __init__(self):
        # counter of storage ids to live references
        self.live_storages = weakref.WeakKeyDictionary()
        self.memory_use = 0
        self.max_memory = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        rs = func(*args, **kwargs)
        breakpoint()
        tree_map_only(torch.Tensor, self.track_tensor_memory_use, rs)
        return rs

    def track_tensor_memory_use(self, tensor):
        # already accounted for
        stor = tensor.untyped_storage()
        if stor in self.live_storages:
            return

        self.live_storages[stor] = tensor.shape
        nbytes = tensor.untyped_storage().nbytes()

        # new storage, add to memory
        self.change_memory(nbytes)

        # when this storage dies, we need to adjust memory
        weakref.finalize(stor, functools.partial(self.change_memory, -nbytes))

    def change_memory(self, delta):
        self.memory_use += delta
        self.max_memory = max(self.memory_use, self.max_memory)
    
MB = 2 ** 20
GB = 2 ** 30


def fn(batch_size):
    print(f"Running batch size {batch_size}")
    
    with FakeTensorMode(), enable_python_dispatcher():
        with TensorMemoryProfilerMode() as tmpm:
            # resnext50_32x4d = models.resnext50_32x4d(pretrained=False)
            model = torch.nn.Linear(1024, 2048, bias=False, device="cuda", dtype=torch.float32)
            input = torch.randn(batch_size, 1024, dtype=torch.float32, device="cuda", requires_grad=True)

            output = model(input)
            print(f"GB after forward: {tmpm.max_memory / GB}")
            print(f"After forward: {list(tmpm.live_storages.keys())=}")

            output.sum().backward()
            print(f"GB after backward: {tmpm.max_memory / GB}")
            print(f"After backwards: {list(tmpm.live_storages.keys())=}")
            return tmpm.max_memory

for i in range(10, 50, 10):
    fn(i)