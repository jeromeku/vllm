# import weakref, gc

# class Payload:
#     def __init__(self, name):
#         self.name = name
#     def __repr__(self):
#         return f"Payload({self.name})"

# # --- ordinary dict ----------------------------------------------------------
# regular = {}
# a = Payload("a")
# regular["a"] = a

# del a                # delete the last *named* reference
# gc.collect()         # force a garbage-collection cycle

# print("regular keys:", list(regular.keys()))   # 👈 still there!

# # --- weak-key dict ----------------------------------------------------------
# weak = weakref.WeakKeyDictionary()
# b = Payload("b")
# weak["b"] = b

# del b
# gc.collect()

# print("weak keys:   ", list(weak.keys()))      # 👈 now empty
import torch, weakref, gc
KB = 2 ** 10
GB = KB ** 3

assert GB == 2 ** 30

aux = weakref.WeakKeyDictionary()          # mapping: tensor ➔ statistics
# aux = dict()
def make_tensor():
    t = torch.ones(2 * GB, dtype=torch.uint8, device="cuda")
    aux[t] = True     # annotate
    return t                               # caller gets the tensor
print(torch.cuda.memory_summary(abbreviated=True))
t = make_tensor()          # caller holds a reference
print("aux has entry?", len(aux))          # -> 1
print(torch.cuda.memory_summary(abbreviated=True))
del t                      # caller drops reference
torch.cuda.empty_cache()   # just to underline the effect
gc.collect()

print("aux cleaned up? ", len(aux))        # -> 0
print(torch.cuda.memory_summary(abbreviated=True))
