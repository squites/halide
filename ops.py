from ir import Call, Variable, Input, Add, Sub, Mul, Max, ConstInt, ConstFloat, Func, Var, RDom, CallType
#from tensor import Tensor

def op_add(op0, op1): # a,b are Tensors
    # implicit apply broadcast if needed
    # check len(shapes) to see if there's different shapes for broadcasting:
    vars = [Var(f'i{n}') for n in range(len(op0.shape))]
    args = [Variable(n.name) for n in vars]
    a = Call(name=op0.func.name, args=args, ref=op0.func, call_type=CallType.INPUT)
    b = Call(name=op1.func.name, args=args, ref=op1.func, call_type=CallType.INPUT)
    operation = Add(a, b)
    func = Func(name=f'add_{op0.func.name}_{op1.func.name}', args=vars, value=operation, updates=[])
    #return Tensor(func, op0.shape) # this shouldnt return Tensor because it'll have a circular depency with tensor.py
    return (func, op0.shape)

def op_sub(op0, op1): # op0(3,4), op1(3,)
    # need to identify if there's need to bcast and do it implicitly
    # TODO: broadcast (B,C) - (B,)
    outshape = max(len(op0.shape), len(op1.shape))
    vars = [Var(f'i{n}') for n in range(outshape)]
    args = [Variable(n.name) for n in vars]
    #call0 = CallType.INPUT if isinstance(op0.func, Input) else CallType.FUNC
    #call1 = CallType.INPUT if isinstance(op1.func, Input) else CallType.FUNC
    call0 = callto(op0)
    call1 = callto(op1)
    if op0.shape == op1.shape:
        bargs = args
    elif len(op1.shape) < len(op0.shape):
        # op1=1D, op0=2D
        if op1.shape == op0.shape[:len(op1.shape)]:
            bargs = args[:len(op1.shape)]   # only first N vars
        else:
            raise ValueError(f"unsupported shapes {op0.shape} {op1.shape}")
    #op1.shape == (op0.shape[1],) or op1.shape == (1,): # need to encapsulate in a tuple
        # broadcast
        #bargs = [args[len(args)-1]]
    else:
        raise ValueError(f"unsupported shapes {op0.shape} {op1.shape}")
    a = Call(name=op0.func.name, args=args, ref=op0.func, call_type=call0)
    b = Call(name=op1.func.name, args=bargs, ref=op1.func, call_type=call1)
    func = Func(name=f'sub_{op0.func.name}_{op1.func.name}', args=vars, value=Sub(a, b), updates=[])
    return (func, op0.shape)

def op_mul(op0, op1):
    i = Var("i")                                                                               # 1-declare the coordinate
    a = Call(name=op0.func.name, args=[Variable("i")], ref=op0.func, call_type=CallType.INPUT) # 2-read 'a' at index 'i'
    b = Call(name=op1.func.name, args=[Variable("i")], ref=op1.func, call_type=CallType.INPUT) # 3-read 'b' at index 'i'
    operation = Mul(a, b)                                                                      # 4-scalar math 
    func = Func(name=f'mul_{op0.func.name}{op1.func.name}', args=[i], value=operation, updates=[]) # 5-wrap back to Tensor 
    #return Tensor(func, op0.shape)
    return (func, op0.shape)

def op_div(op0, op1):
    vars = [Var(f'i{n}') for n in range(len(op0.shape))]
    args = [Variable(n.name) for n in vars]
    call = CallType.INPUT if isinstance(op0.func, Input) else CallType.FUNC
    a = Call(name=op0.func.name, args=args, ref=op0.func, call_type=call)
    b = Call(name=op1.func.name, args=args, ref=op1.func, call_type=call)
    op = Div() # actually I can implement with Mul() instead of div
    func = Func(name=f'div_{op0.func.name}_{op1.func.name}', args=vars, value=op, updates=[])
    return (func, op0.shape)

def op_sum(op0, axis=0):
    # TODO: implement sum over axis.
    # op0 = Tensor(Func("add", [Var("i")], Add(Call("in0", [Variable("i")], ref=Input("in0", 1, int32), call_type=INPUT), Call("in1", [Variable("i")], ref=Input("in1", 1, int32), call_type=INPUT))), shape=(4,))))
    reduce = RDom(name="rloop", min=0, extent=op0.shape[axis]) # Im saying that I created a Reduction Loop variable named "rloop" that'll range from 0 to 4 (excluding 4) in the case of shape=(4,)
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
    #return Tensor(func=f, shape=())
    return (f, ())

def op_matmul(M, N):
    # M -> Tensor(Input("name=in0", shape=(3,2), dtype=int32))
    # N -> Tensor(Input("name=in1", shape=(2,3), dtype=int32))
    i = Var("i")
    j = Var("j")
    k = RDom(name="k", min=0, extent=M.shape[1]) # k loops over the inner dim "M.shape[1]"
    call0 = CallType.INPUT if isinstance(M.func, Input) else CallType.FUNC # use this instead of hardcode 
    call1 = CallType.INPUT if isinstance(N.func, Input) else CallType.FUNC
    m0 = Call(name=M.func.name, args=[Variable("i"), Variable("k")], ref=M.func, call_type=call0) # read M at index [i][k]
    n0 = Call(name=N.func.name, args=[Variable("k"), Variable("j")], ref=N.func, call_type=call1) # read N at index [k][j]
    # matmul is a reduction, so Mul happens inside the fold. 
    # C[i][j] = 0
    f = Func(name=f'matmul_{M.func.name}_{N.func.name}', args=[i, j], value=ConstInt(0), updates=[])    
    self_call = Call(name=f'matmul_{M.func.name}_{N.func.name}', args=[Variable("i"), Variable("j")], ref=f, call_type=CallType.FUNC) # acc
    update = ([Variable("i"), Variable("j")], Add(self_call, Mul(m0, n0)), k)
    f.updates.append(update)
    return (f, (M.shape[0], N.shape[1]))

def op_relu(t0):
    #i = Var("i")
    #j = Var("j")
    vars = [Var(f'i{n}') for n in range(len(t0.shape))]
    variables = [Variable(n.name) for n in vars]
    call = CallType.INPUT if isinstance(t0.func, Input) else CallType.FUNC
    #x = Call(name=t0.func.name, args=[Variable("i"), Variable("j")], ref=t0.func, call_type=call)
    x = Call(name=t0.func.name, args=variables, ref=t0.func, call_type=call)
    #f = Func(name=f'relu_{t0.func.name}', args=[i, j], value=Max(ConstInt(0), x), updates=[])
    f = Func(name=f'relu_{t0.func.name}', args=vars, value=Max(ConstInt(0), x), updates=[])
    return (f, t0.shape)

def op_exp(t0):
    vars = [Var(f'i{n}') for n in range(len(t0.shape))]
    variables = [Variable(n.name) for n in vars]
    call = CallType.INPUT if isinstance(t0.func, Input) else CallType.FUNC
    x = Call(name=t0.func.name, args=variables, ref=t0.func, call_type=call)
    ext = Call(name="expf", args=[x], ref=None, call_type=CallType.EXTERN)
    f = Func(name=f'exp_{t0.func.name}', args=vars, value=ext, updates=[])
    return (f, t0.shape)

# verify this sum() version
def op_sum(t0, axis=0):
    # example: Tensor(Func(...), shape=(3,3))
    # for i0 in range(t0.shape[0]):
    #   out[i0] = -inf
    #   for r in range(t0.shape[1]):
    #       out[i0] = Sum(out[i0], t0[i0, r])
    # kept(outdims): is the output shape (the dims that will remain the same after reduce) which is exactly the shape of the out tensor. Because its reducing, the axis dim will be reduced to a scalar, dropping that dim completely 
    outdims = [n for n in range(len(t0.shape)) if n != axis]
    vars = [Var(f'i{n}') for n in range(len(outdims))]
    args = [Variable(v.name) for v in vars]
    args2 = args.copy() # to get [Variable(..)] list without RDom var.
    # reduce is the RDom variable that'll sweep from 0..t0[axis]. if axis=1 from 0 to rowsize
    reduce = RDom(name="r", min=0, extent=t0.shape[axis])
    args.insert(axis, Variable(reduce.name))
    call = callto(t0.func)
    # x is the element being folded in this iteration. So is the current value being compared with the acc (acc could be max, could be sum)
    # Call is to tell the compiler where to read this element (t6[i0, r]). So read from "t6" at coordinates [i0,r]
    x = Call(name=t0.func.name, args=args, ref=t0.func, call_type=call)
    # The output tensor must exist before anything writing/reading at this tensor.
    # here we declare the tensor, with args which are the vars/coordinates of the output tensor which were determined by `kept/outshape` 
    # value is the initial value of the tensor before the updates
    # updates is temporarily empty
    f = Func(name=f'sum_{t0.func.name}', args=vars, value=ConstInt(0), updates=[])
    self_f = Call(name=f.name, args=)

    return (f, shape)

def op_maxreduce(t0, axis=0): # axis tells which dim it wants to reduce. (2,4) axis=1, then reduce dim with 4, so softmax on each row independently
    # axis=0 max_reduce each row of a col
    # axis=1 max_reduce each col of a row
    # maxreduce works like this:
    # for i0 in range(t0.shape[0]):
    #   out[i0] = -inf  
    #   for r in range(t0.shape[1]):
    #       out[i0] = Max(out[i0], t0[i0, r])

    kept = [n for n in range(len(t0.shape)) if n != axis]  # [0, 1, ...] # this n!=axis
    vars = [Var(f'i{n}') for n in range(len(kept))]        # [Var("i0"), Var("i1")]
    args = [Variable(v.name) for v in vars]                # [Variable("i0"), Variable("i1")]
    args2 = args.copy()                                    # [Variable("i0"), Variable("i1")]
    # loop header
    reduce = RDom(name="r", min=0, extent=t0.shape[axis])
    args.insert(axis, Variable("r"))
    call = CallType.INPUT if isinstance(t0.func, Input) else CallType.FUNC
    # axis=1 need row coordinate
    x = Call(name=t0.func.name, args=args, ref=t0.func, call_type=call)
    # init func
    f = Func(name=f'maxreduce_{t0.func.name}', args=vars, value=ConstFloat(float("-inf")), updates=[])
    self_call = Call(name=f'maxreduce_{t0.func.name}', args=args2, ref=f, call_type=CallType.FUNC)
    update = (args2, Max(self_call, x), reduce)
    f.updates.append(update)
    shape = tuple(t0.shape[n] for n in kept)
    return (f, shape)

    # [1, 2, 3]     [3] 
    # [4, 5, 6] ->  [6] -> max of each row. axis=1, because checks over cols of each row to find the max
    # [9, 8, 1]     [9]

    # t0.shape(2,4)
    # len(t0.shape) = 2
    # AXIS=1                                                AXIS=0
    # kept=[0] if axis=1;                                   kept=[1] if axis=0; So removes the index of the axis
    # vars=[Var("i0")] if axis=1;                           vars=[Var("i1")] if axis=0
    # args=[Variable("i0")] if axis=1;                      args=[Variable("i1")] if axis=0
    # insert=[Variable("i0"), Variable("r")] if axis=1      insert=[Variable("r"), Variable("i1")] if axis=0
    # Call("input", [Variable("i0"), Variable("r")]...)     Call("input", [Variable("r"), Variable("i1")]...)
    # Func("x", [Var("i0")], -inf,...)                      Func("x", [Var("i1")], -inf)
    # Call("x", [Variable("i0")], ...)                      Call("x", [Variable("i1")], ...)
    # ([Variable("i0")], Max(), RDom())                     ([Variable("i1")], Max(), RDom)


def op_softmax(t0):
    from tensor import Tensor
    # mx = max(t0)
    # x = exp(t0 - mx) -> bcast
    # return x / sum(x)
    # Essentially:
    # exp(t0 - max(t0)) / sum(exp(t0 - max(t0))) 

    max = Tensor(*op_maxreduce(t0, axis=len(t0.shape)-1)) # Tensor(Func(...), shape)
    #max = op_maxreduce(t0, axis=len(t0.shape)-1)
    print("mx:", max)
    sub = Tensor(*op_sub(t0, max))
    print("sub:", sub)
    exp = Tensor(*op_exp(sub))
    print("exp:", exp)
    sum = Tensor(*op_sum(exp)) # not working yet, i need to do sum over axis
    print("sum:", sum)
    
    div = op_div(exp, sum)
    return (t0.func, t0.shape)
    #sub = Tensor(op_sub(t0, mx)) # Tensor(Input(...)) - Tensor(Func(...)) # needs to bcast inside sub
    #x = Tensor(op_exp(sub))
    #sum = Tensor(op_sum(x))
    #f = op_div(x, sum)
    #return (mx, t0.shape)
    #return (f, t0.shape)

def callto(x):
    if isinstance(x.func, Input):
        return CallType.INPUT
    else:
        return CallType.FUNC


"""
a: [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]            shape (3, 3)

b: [10, 20, 30]           shape (3,)

align from the right
a: (3, 3)
b: (   3)   ← only one dimension, so it aligns with the RIGHT 3

So b is treated as if it has shape (1, 3) — one row, three columns. That single row is reused for all 3 rows of a:

a:          b (conceptually):      result:
[1, 2, 3]   [10, 20, 30]          [1+10, 2+20, 3+30]  = [11, 22, 33]
[4, 5, 6] + [10, 20, 30]        = [4+10, 5+20, 6+30]  = [14, 25, 36]
[7, 8, 9]   [10, 20, 30]          [7+10, 8+20, 9+30]  = [17, 28, 39]


Example 2 (softmax):
a: [10, 20, 30, 40]     shape (4,)
b: 5                    shape () — just one number

result = a - b
       = [10-5, 20-5, 30-5, 40-5]
       = [5, 15, 25, 35]
The scalar 5 is reused for every element

"""