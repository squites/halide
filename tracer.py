from typing import Optional, Union
from enum import Enum, auto
from dataclasses import dataclass

# move elsewhere
class CallType(Enum):
    INPUT = auto()
    FUNC = auto()

# Hierarchy 1 - Expressions
class Expr:
    """ base class """
    pass 

@dataclass
class Call(Expr): # Call is "read one element of that thing (name), at these coordinates"
    name: str        # thing
    args: list[Expr] # coordinates
    ref: Union["Func","Input"] # ptr to the thing
    call_type: CallType #str # tag saying what kind of thing the "name" refers to.

    def __repr__(self) -> str:
        return f'Call({self.name}, {self.args}, {self.call_type})'

@dataclass
class Variable(Expr): # reference by name to a Var, RVar, or Let binding
    name: str

    def __repr__(self) -> str:
        return f'Variable({self.name})'

# ops
@dataclass
class Add(Expr):
    op0: Expr
    op1: Expr

    def __repr__(self) -> str:
        return f'Add({self.op0}, {self.op1})'

@dataclass
class Mul(Expr):
    op0: Expr
    op1: Expr

    def __repr__(self) -> str:
        return f'Add({self.op0}, {self.op1})'

# Remove!
@dataclass
class IntImm(Expr):
    value: int

@dataclass
class ConstInt(Expr): # substitute IntImm for this one. more intuitive
    value: int

    def __repr__(self) -> str:
        return f'ConstInt({self.value})'


# hierarchy 2 - Structure
class Func: # represent a tensor or computation stage. has name, dims, 
    def __init__(self, name: str, args: list["Var"], value: Expr, updates: Optional[list[tuple]]):
        self.name = name
        self.args = args # indices that the computation will operate (i,j)
        self.value = value
        self.updates = updates # [(lhs_args, rhs, rdom)]
        #self.schedules = schedules

    def __repr__(self) -> str:
        return f'Func({self.name}, {self.args}, {self.value}, {self.updates})'

class Input: # external input, previously ImageParam from halide's
    def __init__(self, name, dims, type):
        self.name = name
        self.dims = dims
        self.type = type

    def __repr__(self) -> str:
        return f'Input({self.name}, {self.dims}, {self.type})'

class Var: # just a name
    def __init__(self, name):
        self.name = name
    
    def __repr__(self) -> str:
        return f'Var({self.name})'

class RDom:
    def __init__(self, name: str, min: Expr, extent: Expr): # I dont like extent. maybe change to 'range' or 'max'
        self.name = name
        self.min = min
        self.extent = extent

    def __repr__(self) -> str:
        return f'RDom({self.name}, {self.min}, {self.extent})'

# wrapper. move to other file  
class Tensor:
    def __init__(self, func: Union[Func, Input], shape: tuple):
        # 'func' is a reference to either data-to-come(input) or a recipe(func)
        self.func = func # an array defined by a rule (Func) or an array arriving in runtime (Input). Also the 'ref'
        self.shape = shape

    @staticmethod
    def input(shape, dtype="int", name="in"): # this is how the user declares "data will arrive here at runtime"
        return Tensor(Input(name, dims=len(shape), type=dtype), shape)

    def __repr__(self) -> str:
        return f'Tensor({self.func!r}, {self.shape!r})'

    def __add__(self, other):
        return op_add(self, other)

    def __mul__(self, other):
        return op_mul(self, other)
    
    def sum(self):
        return op_sum(self)

def op_add(op0, op1): # a,b are Tensors
    # verify that a.shape and b.shape are equal? if (a.shape != b.shape) broadcast? error? ignore?
    i = Var("i") # the coordinate. The variable that output depends on. Output shape is (4,) -> 1 dim -> 1 Var -> i = Var("i")
    a = Call(name=op0.func.name, args=[Variable("i")], ref=op0.func, call_type=CallType.INPUT)
    b = Call(name=op1.func.name, args=[Variable("i")], ref=op1.func, call_type=CallType.INPUT)
    operation = Add(a, b)
    func = Func(name=f'add_{op0.func.name}_{op1.func.name}', args=[i], value=operation, updates=[])
    return Tensor(func, op0.shape)

def op_mul(op0, op1):
    i = Var("i")                                                                               # 1-declare the coordinate
    a = Call(name=op0.func.name, args=[Variable("i")], ref=op0.func, call_type=CallType.INPUT) # 2-read 'a' at index 'i'
    b = Call(name=op1.func.name, args=[Variable("i")], ref=op1.func, call_type=CallType.INPUT) # 3-read 'b' at index 'i'
    operation = Mul(a, b)                                                                      # 4-scalar math 
    func = Func(name=f'mul_{op0.func.name}{op1.func.name}', args=[i], value=operation, updates=[]) # 5-wrap back to Tensor 
    return Tensor(func, op0.shape)

def op_sum(op0):
    # op0 = Tensor(Func("add", [Var("i")], Add(Call("in0", [Variable("i")], ref=Input("in0", 1, int32), call_type=INPUT), Call("in1", [Variable("i")], ref=Input("in1", 1, int32), call_type=INPUT))), shape=(4,))))
    reduce = RDom(name="rloop", min=0, extent=op0.shape[0]) # Im saying that I created a Reduction Loop variable named "rloop" that'll range from 0 to 4 (excluding 4) in the case of shape=(4,)
    x = Call(name=op0.func.name, args=[Variable("rloop")], ref=op0.func, call_type=CallType.FUNC) # Im saying Call object represents that'll read one element from the function "op0.func.name" at the coordinate "rloop". 
    # the Func needs to exist before acc so acc.ref points to it
    f = Func(name=f'sum', args=[], value=ConstInt(0), updates=[])
    # here this is reading whatever sum produces, which is the accumulator scalar
    acc = Call(name=f'sum', args=[], ref=f, call_type=CallType.FUNC) # this Call() is a call to the sum function recursive
    # reduction will operate over `acc += add[rloop]` for example
    body = Add(acc, x) # accumulator + x
    # ([], Add(acc,x), RDom("rloop", [0..3])). "lhs" is empty because is a scalar value
    update = ([], body, reduce)
    f.updates.append(update) 
    #f = Func(name=f'sum', args=[], value=ConstInt(0), updates=updates) # 'value' here is the initial value
    return Tensor(func=f, shape=())

# ref is a pointer to the Func or Input being called. ex:
# a=Tensor(Input("in0", 1, int32), (4,))
# b=Tensor(Input("in1", 1, int32), (4,))
# Tensor(Input("in0", 1, int32), (4,)) + Tensor(Input("in1", 1, int32), (4,))
# left = Call("in0", [Variable("i")], ref=Input("in0", 1, int32), call_type="INPUT")
# right = Call("in1", [Variable("i")], ref=Input("in1", 1, int32), call_type="INPUT")
# Tensor(Func("add", [idx], Add(left, right)))


# main
a = Tensor.input((4,), name="in0") # Tensor(Input("in0", 1, int32), (4,))
b = Tensor.input((4,), name="in1") # Tensor(Input("in1", 1, int32), (4,))
c = (a + b)                        # Tensor(Func("add0", [Var("i")], Add(Call("in0", [Variable("i")], ref=Input("in0", 1, int32), call_type=INPUT), Call("in1", [Variable("i")], ref=Input("in1", 1, int32), call_type=INPUT))), shape=(4,))
out = c.sum() # implement sum()

#print("c: ", c.__dict__)
#for i,j in (c.func.__dict__).items():
#    print(f'{i}: {j}')
#
#print("\nSUM:")
#for i,j in (out.func.__dict__).items():
#    print(f'{i}: {j}')

print(out)


# What Call() actually means:
# 'Call' node means: "read 'name' at these coordinates". and there are 3 things Call can reads from:
    #   Call("matmul0", [i,j], FUNC) - a Func(). Compiler find its definition, schedule it, generate its loops, wire up producer→consumer. This is a DAG edge.
    #   Call("a", [i,k], INPUT) - an Input(). No definition exists; this becomes a memory load from a runtime buffer.
    #   Call("expf", [x], EXTERN) - external C function. codegen emits a literal function call

# What updates[(,)] means:
# updates = [(lhs_args, rhs, rdom)] means: "For every value in rdom, evaluate the scalar expression rhs, 
# and store it at the output coordinates given by lhs_args."
#
# lhs_args is not the destination buffer itself. The destination is always implicit: 
# it's the Func that owns this updates list.
# lhs_args contains the index expressions that determine where inside that Func to write. 
# So yes, they are Variable (or other Expr) nodes — because they are coordinates, not the buffer name.


""" 
Obs: This example from the paper:
`   UniformImage in(UInt(8), 2)
    RDom r(0..in.width(), 0..in.height()), ri(0..255)
    Var x, y, i
    Func histogram(i) = 0; histogram(in(r.x, r.y))++
    Func cdf(i) = 0; cdf(ri) = cdf(ri-1) + histogram(ri)
    Func out(x, y) = cdf(in(x, y))
`
is something that the user writes. Halide is a language, so the user represents the algorithm this way. My plan
is to trace it, so the model will be represented similar to this, but not how this is written, but directly
the DAG.

Buffer: represent input tensor. has name, dtype, shape. Buffer is a declaration "there is a input tensor called a, ...."
Var: is a coordinate. Referencing position 'i' in something
Expr: expression tree is how you describe math symbolicaly "a(i) + b(i)" -> Add(Call(Buffer("a"), idx=Var("i")), Call(Buffer("b"), idx=Var("i")))

IR:
# expressions — always about ONE element
Expr
├─ IntImm(value) / FloatImm(value)
├─ Variable(name)                          # mention of a coordinate
├─ Add(a, b) / Mul(a, b) / ...             # scalar math
└─ Call(name, ref, args, call_type)        # read ONE element of ref at args

# structure — not expressions
Var(name)                                  # declare a pure coordinate
RDom(name, min, extent)                    # declare an ordered loop coordinate
Input(name, dims, dtype)                   # runtime array, no definition
Func(name, args, value, updates)           # array defined by a rule
   updates: [(lhs_args, rhs, rdom)]        # a parked loop: C-line, B-line
Tensor(func, shape)                        # user handle, never inside IR


VERY IMPORTANT OBS:
if you think about the objects, if its a simple Var("i"), or if its a RDom("r",...). 
They are pretty much equal in the sense that they simply will be passed as coordinates in 
form of args to the function. The only difference is that the compiler will need 
to understand that in the case of RDom the scalars on each "r" (loop variable) coordinate will eventually 
reduce, and the Var wont. So to complete the understanding every function operates and result a single scalar. 
So Add() operating into Var coordinate, it will apply the expression on each coordinate over the tensor, 
where each coordinate results into a value. In the case of a reduction operation like sum, the same happens
 where the expression is done on each coordinate "r", but these values are being accumulated.
The Func() of that expression is the one that produces the vector in this case of "add". add0[0..3], each
element one scalar addition 

# This is the RECIPE — not an execution
body = Add(
    Call("in0", [Variable("i")]),   # template: "read in0 at current i"
    Call("in1", [Variable("i")])    # template: "read in1 at current i"
)

# This says: "evaluate the recipe for i = 0, 1, 2, 3 for shape(4,)"
f = Func("add0", args=[i], value=body)

At compile time, there is one Call node in your IR tree.
At runtime, the generated code evaluates that same expression 4 times (once per value of i):

// Generated code — the Func becomes this loop
for (int i = 0; i < 4; i++) {
    add0[i] = in0[i] + in1[i];   // ONE scalar Add per iteration
}

So yes — each iteration produces one scalar. But the single Call node in the IR generates 4 memory loads 
at runtime because it sits inside a Func whose domain has 4 points.


``` 
"""
# --------------------------------------------------------------------------------------------------------
#def walk(node, topo, visited=None):
#    """ walk the graph """
#    if visited is None:
#        visited = set()
#
#    if node.name in visited:
#        return node
#    visited.add(node.name)
#
#    for i in node.inputs:
#        walk(i, topo, visited)
#    topo.append(node)
#    return node
#
## oracle? remove this later
#def check_matmul(a, b, c, M, N, K):
#    for i in range(M):
#        for j in range(N):
#            tmp = 0
#            for k in range(K):
#                tmp += a[i][k] * b[k][j]
#            c[i][j] = tmp


"""
take a small neural network (even just a matmul + relu), trace it into a graph IR, 
implement fusion (merge that matmul + relu into one kernel), 
implement tiling (break the matmul into blocks that fit in shared memory), 
generate actual GPU code, and run it. That one project forces you through every concept 
— graph representation, 
- scheduling, 
- the fusion decision problem, 
- memory hierarchy, 
- code generation

there are 4 levels of IR:
1) Graph IR (Tensor-op level):
    Reason about the operations and dataflow.
    - op fusion, constant folding, dead code elim, algebraic simplification,...

2) Loop IR (loop-nest lvl):
    ops are expanded into explicit loops over memory. reason about loops, iteration order, mem. access...
    - loop tiling, interchange, unrolling, buffer/mem planning, ...

3) renderer (kernel-lang lvl):
    basically walk Loop IR, and generate cuda:
    - thread/block mapping, smem staging, mem-access coalescing, ..

4) ISA (PTX/llvm IR):
    - reg. allocation, instruction scheduling, peephole optimization, ...






i = Var("i")                                          # 1) declare the coordinate
    left = Call(a.func.name, [Variable("i")], "INPUT")    # 2) read 'a' at index 'i'
    right = Call(b.func.name, [Variable("i")], "INPUT")   # 3) read 'b' at index 'i'
    body = Add(left, right)                               # 4) scalar math
    f = Func("add0", args=[i], value=body)                # 5) wrap into Tensor definition
    return Tensor(func=f, shape=a.shape)

"""