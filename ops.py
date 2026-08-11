from ir import Call, Variable, Input, Add, Mul, Max, ConstInt, ConstFloat, Func, Var, RDom, CallType
#from tensor import Tensor

def op_add(op0, op1): # a,b are Tensors
    # verify that a.shape and b.shape are equal? if (a.shape != b.shape) broadcast? error? ignore?
    i = Var("i") # the coordinate. The variable that output depends on. Output shape is (4,) -> 1 dim -> 1 Var -> i = Var("i")
    a = Call(name=op0.func.name, args=[Variable("i")], ref=op0.func, call_type=CallType.INPUT)
    b = Call(name=op1.func.name, args=[Variable("i")], ref=op1.func, call_type=CallType.INPUT)
    operation = Add(a, b)
    func = Func(name=f'add_{op0.func.name}_{op1.func.name}', args=[i], value=operation, updates=[])
    #return Tensor(func, op0.shape) # this shouldnt return Tensor because it'll have a circular depency with tensor.py
    return (func, op0.shape)

def op_mul(op0, op1):
    i = Var("i")                                                                               # 1-declare the coordinate
    a = Call(name=op0.func.name, args=[Variable("i")], ref=op0.func, call_type=CallType.INPUT) # 2-read 'a' at index 'i'
    b = Call(name=op1.func.name, args=[Variable("i")], ref=op1.func, call_type=CallType.INPUT) # 3-read 'b' at index 'i'
    operation = Mul(a, b)                                                                      # 4-scalar math 
    func = Func(name=f'mul_{op0.func.name}{op1.func.name}', args=[i], value=operation, updates=[]) # 5-wrap back to Tensor 
    #return Tensor(func, op0.shape)
    return (func, op0.shape)

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

def op_maxreduce(t0, axis=0): # axis tells which dim it wants to reduce. (2,4) axis=1, then reduce dim with 4, so softmax on each row independently
    # axis=0 max_reduce each row of a col
    # axis=1 max_reduce each col of a row
    kept = [n for n in range(len(t0.shape)) if n != axis]  # [0, 1, ...] # this n!=axis 
    vars = [Var(f'i{n}') for n in range(len(kept))]        # [Var("i0"), Var("i1")]
    args = [Variable(v.name) for v in vars]                # [Variable("i0"), Variable("i1")]
    args2 = args.copy()                                    # [Variable("i0"), Variable("i1")]
    args.insert(axis, Variable("r"))                       # [Variable("i0"), Variable("i1"), Variable("r")] # if axis=0, then variables[r, i1]
    # loop header
    reduce = RDom(name="r", min=0, extent=t0.shape[axis])
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
    #
    # 1 2 3                                                              3
    # 2 2 5 -> returns a vector where each pos is the max of row/col ->  5
    # 7 1 0             (this is reducing on axis=1)                     7
    #
    # axis=1 means we're reducing the 2nd dim, so in a tensor (2,4), we're reducing on 4, which is each column
    # of a row.


