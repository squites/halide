from tensor import Tensor
from ops import Func, find_direct_calls, find_transitive_calls, calldfs, ENV_TABLE

# tracer file will be to load an example or a model from extern places
#print("\n --- MLP --- ")
#x  = Tensor.input((8, 16), name="x")     # batch 8, features 16
#w1 = Tensor.input((16, 32), name="w1")
#b1 = Tensor.input((8,), name="b1")
#w2 = Tensor.input((32, 10), name="w2")
#h  = (x @ w1) + b1                           # (8,32)  matmul
##h  = (h + b1)                            # broadcast add — careful, see note
#h  = h.relu()
#o  = (h @ w2)                            # (8,10)  second matmul
#p  = o.softmax()                         # 5 more Funcs
#
## erase everything and only call realize()
#print("\nENV_TABLE")
#for k,v in ENV_TABLE.items():
#    print(f'** {k}: {v}')
#
## realize
#p.func.realize(p.shape)
#
#print(" --- before schedule: ", p.func.schedules)
## write schedule rule
#p.func.compute_root()
## check if schedule recorded
#print(" --- after schedule: ", p.func.schedules)


# Write the schedules here:
# e.g.: g.compute_root(), g.compute_at(out, x)

# Simpler example:
A = Tensor.input((3,2), name="A")
B = Tensor.input((3,2), name="B")
f = A + B
out = f + A
print("OUT:\n", out)

# test schedule
f.func.compute_root()
out.func.compute_root()

out.func.realize(out.shape)
print("before: ", out.func.schedules)
#out.func.compute_root()
#print("after: ", out.func.schedules)




"""
FUTURE IDEA:
* auto scheduler (cost model / heuristics)
* agents decides the schedules, learn with them? (agents/compiler 2.0)
"""