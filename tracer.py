from tensor import Tensor
from ops import find_direct_calls, find_transitive_calls, toposort, dfs, MAP

# tracer file will be to load an example or a model from extern places
# move to examples
##a = Tensor.input((4,), name="in0") # Tensor(Input("in0", 1, int32), (4,))
##b = Tensor.input((4,), name="in1") # Tensor(Input("in1", 1, int32), (4,))
##c = (a + b)                        # Tensor(Func("add0", [Var("i")], Add(Call("in0", [Variable("i")], ref=Input("in0", 1, int32), call_type=INPUT), Call("in1", [Variable("i")], ref=Input("in1", 1, int32), call_type=INPUT))), shape=(4,))
##out = c.sum(axis=0) # implement sum()
##print("Sum:")
##print(out)
##
##t0 = Tensor.input((3,2), name="M")
##t1 = Tensor.input((2,3), name="N")
##t2 = t0 @ t1
##print("Matmul:")
##print(t2)
##
##t3 = t2.relu()
##print("Relu:")
##print(t3)
##
##t4 = t3.maxreduce()
##print("Maxreduce:")
##print(t4)
##
##print("\nMaxreduce:")
##t5 = Tensor.input((2,4), name="t5")
##out = t5.maxreduce(axis=1)
##print(out)
##
##print("\nSoftmax:")
##t6 = Tensor.input((3, 4), name="t6")
##print("t6: ", t6)
##t7 = t6.softmax()
##print("\nsoftmax t7: ", t7)

#print("\nMatmul + ReLU + Softmax:")
#t0 = Tensor.input((3,2), name="M")
#t1 = Tensor.input((2,3), name="N")
#t2 = (t0 @ t1).relu().softmax()
#print(t2)
#print("\nMAP:")
#for k,v in MAP.items():
#    print(f'** {k}: {v}')
#calls = find_direct_calls(t2.func)
#print("\nCALLS:")
#for c in calls:
#    print("* ", c)
#print("Transitive:")
#find_transitive_calls(t2.func)

#print(" --- Testing Transitive closure")
#print("Add + Mul + Add + Mul + Add")
#a = Tensor.input((2, 2))
#b = Tensor.input((2, 2))
#x = a + b
#y = x * b
#z = x + y
#w = z * a
#out = w + y

#find_transitive_calls(out.func)

print("\n --- MLP --- ")
x  = Tensor.input((8, 16), name="x")     # batch 8, features 16
w1 = Tensor.input((16, 32), name="w1")
b1 = Tensor.input((8,), name="b1")
w2 = Tensor.input((32, 10), name="w2")
h  = (x @ w1)                            # (8,32)  matmul
h  = (h + b1)                            # broadcast add — careful, see note
h  = h.relu()
o  = (h @ w2)                            # (8,10)  second matmul
p  = o.softmax()                         # 5 more Funcs

print("\nMAP")
for k,v in MAP.items():
    print(f'** {k}: {v}')

print("\nCALLS")
direct = find_direct_calls(p.func)
for c in direct:
    print("* ", c)

print("\nTRANSITIVE")
rev = find_transitive_calls(p.func)

print("\nCLEAN MAP")
print("NODE: ", p.func.name)
for k,v in rev.items():
    print(f'** {k}:{v}')

print("\nTOPOSORT")
#toposort(p.func, rev)
x = dfs(p.func, rev)
print(x)
