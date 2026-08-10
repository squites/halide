from tensor import Tensor

# tracer file will be to load an example or a model from extern places
# move to examples
a = Tensor.input((4,), name="in0") # Tensor(Input("in0", 1, int32), (4,))
b = Tensor.input((4,), name="in1") # Tensor(Input("in1", 1, int32), (4,))
c = (a + b)                        # Tensor(Func("add0", [Var("i")], Add(Call("in0", [Variable("i")], ref=Input("in0", 1, int32), call_type=INPUT), Call("in1", [Variable("i")], ref=Input("in1", 1, int32), call_type=INPUT))), shape=(4,))
out = c.sum() # implement sum()
print("Sum:")
print(out)

t0 = Tensor.input((3,2), name="M")
t1 = Tensor.input((2,3), name="N")
t2 = t0 @ t1
print("Matmul:")
print(t2)