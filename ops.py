from ir import Expr, Call, Variable, Input, Add, Sub, Mul, Div, Max, ConstInt, ConstFloat, Func, Var, RDom, CallType

# given a function name, what is its actual Func definition: ("name": Func(...))
MAP = {}
 
def op_add(op0, op1): # op0,op1 are Tensors
    vars = [Var(f'i{n}') for n in range(len(op0.shape))]
    args = [Variable(n.name) for n in vars]
    if op0.shape == op1.shape:
        bargs = args
    elif len(op1.shape) < len(op0.shape):
        if op1.shape == op0.shape[:len(op1.shape)]: bargs = args[:len(op1.shape)]
        else: raise ValueError(f"unsupported shapes {op0.shape} {op1.shape}")
    else:
        raise ValueError(f"unsupported shapes {op0.shape} {op1.shape}") 
    a = Call(name=op0.func.name, args=args, ref=op0.func, call_type=callto(op0.func))
    b = Call(name=op1.func.name, args=bargs, ref=op1.func, call_type=callto(op1.func))
    name = countname("add")
    func = Func(name=name, args=vars, value=Add(a,b), updates=[])
    MAP[func.name] = func
    return (func, op0.shape)

def op_sub(op0, op1):
    # need to identify if there's need to bcast and do it implicitly
    outshape = max(len(op0.shape), len(op1.shape))
    vars = [Var(f'i{n}') for n in range(outshape)]
    args = [Variable(n.name) for n in vars]
    if op0.shape == op1.shape:
        bargs = args
    elif len(op1.shape) < len(op0.shape):
        # op1=1D, op0=2D
        if op1.shape == op0.shape[:len(op1.shape)]: bargs = args[:len(op1.shape)]   # only first N vars
        else: raise ValueError(f"unsupported shapes {op0.shape} {op1.shape}")
    else:
        raise ValueError(f"unsupported shapes {op0.shape} {op1.shape}")
    a = Call(name=op0.func.name, args=args, ref=op0.func, call_type=callto(op0.func))
    b = Call(name=op1.func.name, args=bargs, ref=op1.func, call_type=callto(op1.func))
    name = countname("sub")
    func = Func(name=name, args=vars, value=Sub(a, b), updates=[])
    MAP[func.name] = func
    return (func, op0.shape)

def op_mul(op0, op1):
    i = Var("i")                                                                                 
    a = Call(name=op0.func.name, args=[Variable("i")], ref=op0.func, call_type=callto(op0.func))
    b = Call(name=op1.func.name, args=[Variable("i")], ref=op1.func, call_type=callto(op1.func))
    operation = Mul(a, b) 
    #func = Func(name=f'mul_{op0.func.name}{op1.func.name}', args=[i], value=operation, updates=[])
    name = countname("mul")
    func = Func(name=name, args=[i], value=operation, updates=[])
    MAP[func.name] = func
    return (func, op0.shape)

def op_div(op0, op1):
    vars = [Var(f'i{n}') for n in range(len(op0.shape))]
    args = [Variable(n.name) for n in vars]
    if op0.shape == op1.shape:
        bargs = args
    elif len(op1.shape) < len(op0.shape):
        if op1.shape == op0.shape[:len(op1.shape)]: bargs = args[:len(op1.shape)]
        else: raise ValueError(f'unsuported shapes {op0.shape} {op1.shape}')
    else:
        raise ValueError(f'unsuported shapes {op0.shape} {op1.shape}')
    a = Call(name=op0.func.name, args=args, ref=op0.func, call_type=callto(op0.func))
    b = Call(name=op1.func.name, args=bargs, ref=op1.func, call_type=callto(op1.func))
    name = countname("div")
    func = Func(name=name, args=vars, value=Div(a, b), updates=[])
    MAP[func.name] = func
    return (func, op0.shape)

def op_matmul(M, N): # Tensor(Input("name=in0", shape=(3,2), dtype=int32))
    i = Var("i")
    j = Var("j")
    k = RDom(name="k", min=0, extent=M.shape[1]) # k loops over the inner dim "M.shape[1]"
    call0 = CallType.INPUT if isinstance(M.func, Input) else CallType.FUNC # use this instead of hardcode 
    call1 = CallType.INPUT if isinstance(N.func, Input) else CallType.FUNC
    m0 = Call(name=M.func.name, args=[Variable("i"), Variable("k")], ref=M.func, call_type=call0) # read M at index [i][k]
    n0 = Call(name=N.func.name, args=[Variable("k"), Variable("j")], ref=N.func, call_type=call1) # read N at index [k][j]
    # matmul is a reduction, so Mul happens inside the fold. 
    # C[i][j] = 0
    #f = Func(name=f'matmul_{M.func.name}_{N.func.name}', args=[i, j], value=ConstInt(0), updates=[])    
    #self_call = Call(name=f'matmul_{M.func.name}_{N.func.name}', args=[Variable("i"), Variable("j")], ref=f, call_type=CallType.FUNC) # acc
    name = countname("matmul")
    f = Func(name=name, args=[i, j], value=ConstInt(0), updates=[])    
    self_call = Call(name=f.name, args=[Variable("i"), Variable("j")], ref=f, call_type=CallType.FUNC) # acc
    update = ([Variable("i"), Variable("j")], Add(self_call, Mul(m0, n0)), k)
    f.updates.append(update)
    MAP[f.name] = f
    return (f, (M.shape[0], N.shape[1]))

def op_relu(t0):
    vars = [Var(f'i{n}') for n in range(len(t0.shape))]
    variables = [Variable(n.name) for n in vars]
    call = callto(t0.func)
    x = Call(name=t0.func.name, args=variables, ref=t0.func, call_type=call)
    #f = Func(name=f'relu_{t0.func.name}', args=vars, value=Max(ConstInt(0), x), updates=[])
    name = countname("relu")
    f = Func(name=name, args=vars, value=Max(ConstInt(0), x), updates=[])
    MAP[f.name] = f
    return (f, t0.shape)

def op_exp(t0):
    vars = [Var(f'i{n}') for n in range(len(t0.shape))]
    variables = [Variable(n.name) for n in vars]
    call = CallType.INPUT if isinstance(t0.func, Input) else CallType.FUNC
    x = Call(name=t0.func.name, args=variables, ref=t0.func, call_type=call)
    ext = Call(name="expf", args=[x], ref=None, call_type=CallType.EXTERN)
    #f = Func(name=f'exp_{t0.func.name}', args=vars, value=ext, updates=[])
    name = countname("exp")
    f = Func(name=name, args=vars, value=ext, updates=[])
    MAP[f.name] = f
    return (f, t0.shape)

def op_sum(t0, axis=0):
    # for i0 in range(t0.shape[0]):
    #   out[i0] = 0
    #   for r in range(t0.shape[1]):
    #       out[i0] = Add(out[i0], t0[i0, r])
    # kept(outdims): is the output shape (the dims that will remain the same after reduce) which is exactly the shape of the out tensor. Because its reducing, the axis dim will be reduced to a scalar, dropping that dim completely 
    outdims = [n for n in range(len(t0.shape)) if n != axis]
    vars = [Var(f'i{n}') for n in range(len(outdims))]
    args = [Variable(v.name) for v in vars]
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
    #f = Func(name=f'sum_{t0.func.name}', args=vars, value=ConstInt(0), updates=[])
    name = countname("sum")
    f = Func(name=name, args=vars, value=ConstInt(0), updates=[])
    acc = Call(name=f.name, args=args[:-1], ref=f, call_type=CallType.FUNC)
    rule = (args[:-1], Add(acc, x), reduce) # args[:-1] to remove reduce variable
    f.updates.append(rule)
    shape = tuple(t0.shape[n] for n in outdims)
    MAP[f.name] = f
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
    #f = Func(name=f'maxreduce_{t0.func.name}', args=vars, value=ConstFloat(float("-inf")), updates=[])
    name = countname("maxreduce")
    f = Func(name=name, args=vars, value=ConstFloat(float("-inf")), updates=[])
    self_call = Call(name=f.name, args=args2, ref=f, call_type=CallType.FUNC)
    update = (args2, Max(self_call, x), reduce)
    f.updates.append(update)
    shape = tuple(t0.shape[n] for n in kept)
    MAP[f.name] = f
    """
    return:
    Func(
        name = "maxreduce",
        args = [Var("i0")],
        value = -inf,
        updates = [
            (
                [Variable("i0")],
                Max(
                    Call(..., [Variable("i0")], Func(..., [Var("i0")], -inf, []), CallType.FUNC),
                    Call(..., [Variable("i0"), Variable("r")], Input(tx, (3,2)), CallType.INPUT)
                ),
                RDom("r", 0, 3)
            )
        ]
    )

    OBS: Var() only appears in Func(). Everywhere else is Variable()
    """
    return (f, shape)

def op_softmax(t0):
    from tensor import Tensor
    # exp(t0 - max(t0)) / sum(exp(t0 - max(t0))) 
    max = Tensor(*op_maxreduce(t0, axis=1))
    sub = Tensor(*op_sub(t0, max))
    exp = Tensor(*op_exp(sub))
    sum = Tensor(*op_sum(exp, axis=1))
    div = Tensor(*op_div(exp, sum))
    return (div.func, div.shape)

def callto(x):
    return CallType.INPUT if isinstance(x, Input) else CallType.FUNC

# adds counter for naming functions
_count = 0
def countname(x):
    global _count
    _count += 1
    return f'{x}{_count}'

# DFS finding all nodes in Expr tree "node"
def walk(node):
    out = [node]
    for child in node.children():
        out.extend(walk(child))
    return out

def dfs(node, map, topo=None, visited=None): # change `map` name to something else
    # IMPORANT: 
    # `map`: lookup table produced by find_transitive_calls. We don't iterate on it, just check if 
    # the callee is there. All Funcs needed from out must be in the table.
    # 3 stages:
    # 1. UNVISITED
    # 2. ON STACK - we're inside this node. calles havent finished yet
    # 3. DONE - fully processed and appended to result list
    assert node.name in map, "call to undefined func"
    print("\nNODE:", node.name)
    if visited is None: visited = set()
    if topo is None: topo = []

    visited.add(node.name)
    print("VISITED:", visited)
    call = find_direct_calls(node)
    for v in call:
        print("loop ", v.name)
        if v.name not in visited:
            dfs(v, map, topo, visited)
    topo.append(node.name)
    return topo

# inside Func node, which other Funcs this node calls to directly?
# returns list of Func objs that the calls refer to
def find_direct_calls(node):
    calls = []
    if isinstance(node, Func):
        for i in walk(node.value):
            if isinstance(i, Call) and i.call_type == CallType.FUNC:
                calls.append(i.ref)

        for lhs, rhs, _ in node.updates:
            for i in lhs:
                for l in walk(i):
                    if isinstance(l, Call) and l.call_type == CallType.FUNC:
                        calls.append(l.ref)
            for r in walk(rhs):
                if isinstance(r, Call) and r.call_type == CallType.FUNC:
                    calls.append(r.ref)
    return calls

# implement warshall's algorithm to find every Func reachable from node Func. 
# This is adding complexity by implementing this way.
# Should be a simple DFS on output node, but I did this way because was my intuitive graph knowledge way 
def find_transitive_calls(node):
    # adjacent matrix
    N = len(MAP.keys())
    adj_matrix = [[1 if row == col else 0 for row in MAP.keys()] for col in MAP.keys()] # init diag with 1s
    idx = {key: i for i, key in enumerate(MAP)}
    reverse_idx = {v: k for k, v in idx.items()}
    for k,v in MAP.items(): # "sum0": Func(sum0,...)
        closure = find_direct_calls(v)
        #print(f"{k}: {closure}")
        for j,v2 in MAP.items():
            if v2 in closure:
                adj_matrix[idx.get(k)][idx.get(j)] = 1

    # if there's 1 means i->k. So we need to find if k has any edges from k to any j
    # so if k->j ([k][j]=1), then i->j ([i][j] = 1)
    for k in range(N):
        for row in range(N):
            if adj_matrix[row][k] == 1: # invariant, can be outside loop col
                for col in range(N):
                    if adj_matrix[k][col] == 1:
                        adj_matrix[row][col] = 1
    
    # we have the first adjacent matrix
    print("\nResult Matrix with Transitive Calls:")
    for r in range(N):
        print(adj_matrix[r])

    reach = MAP.copy()
    for k in range(N):
        if adj_matrix[idx.get(node.name)][k] == 0: # NOT N-1. m[N-1][k], needs to be the idx of the node passed as arg (m[idx.get(node.name)][k])
            reach.pop(reverse_idx[k])
    return reach


"""
a: [[1, 2, 3],
    [7, 8, 9],            shape (3, 3)
    [4, 5, 6]]

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