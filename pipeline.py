# this will have the realize function. So the builder will generate a Stmt tree. It'll receive as 
# input the env table + toposort list of nodes
from ops import find_transitive_calls, calldfs, ENV_TABLE, Add, Mul, Call, CallType
from ir import Func, Realize, Produce, Consume, Provide, For, ForType, Variable

# IMPORTANT: Halide builds a single Stmt tree incrementally. When it processes a function, 
# it constructs that function's computation and injects it into an existing statement tree at 
# the location specified by its schedule. The existing tree becomes the consumer context for the 
# next function.

def visit(node, varmap): # pass only node.value
    # never walk node.ref on Provide
    print("* visiting: ", node)
    if isinstance(node,  (Add, Mul)):
        op0 = visit(node.op0, varmap)
        op1 = visit(node.op1, varmap)
    elif isinstance(node, Call):
        args = [Variable(varmap[i.name]) for i in node.args]
        node.args = args

def build_provide(node, varmap):
    print("\nBUILD PROVIDE: ", node.name)
    args = [Variable(varmap[key]) for key in varmap.keys()]
    visit(node.value, varmap)
    provide = Provide(node.name, args, node.value)
    return provide

# recursive test for build loops
def x(body, node, dim_shape):
    #loop = For(node.name, min=0, extent=shape[0], body=body, fortype=ForType.SERIAL)
    loop = For(node.name, min=0, extent=dim_shape, body=body, fortype=ForType.SERIAL)
    return loop


# make this recursive and remove build_provide from. schedule_functions will call each builder
def build_loopnest(node, shape):
    print("\nLOOP NEST:\n", node)
    # check if its inline
    if node.schedules.compute_lvl == "inline": return

    # rename vars (one dict per Func)
    varmap = {}
    for i in node.args:
        varmap[i.name] = f'{node.name}.{i.name}'

    #args = [Variable(varmap[key]) for key in varmap.keys()]

    provide = build_provide(node, varmap)
    print("\nPROVIDE: \n", provide)

    # building loops
    for i in reversed(range(len(node.schedules.dims))):
        print("i: ", i)
        test = x(provide, node, shape[i])
        provide = test
    print("\nTEST: ", test)



def schedule_functions(env, rev_topo, shape):
    print("\nSCHEDULE FUNCTIONS:")
    # initialize empty stmt tree
    stmt = None

    for i in rev_topo:
        print("\n**", env[i])
        copy = env[i]
        # 1) build Provide() - walk tree renaming vars
        # provide = build_provide(copy)
        # 2) build loop nest
        build_loopnest(copy, shape)


def bounds_inference():
    pass

def storage_flattening():
    pass

def realization_order(out, table):
    return calldfs(out, table)

def lower(out, shape):
    #  a) build_env(): this builds the lookup table of the Funcs. in: out Func. out: env table
    print("\n -- LOWER --")
    print("\nTRANSITIVE")
    env = find_transitive_calls(out)
    print(f"\nENV MAP ({out.name})")
    for k,v in env.items():
        print(f'** {k}:{v}')

    #  b) realization_order(): generates the toposorted dependency. In this code is "call_dfs", change to "realization_order"
    topo = realization_order(out, ENV_TABLE)
    print("ORDER: ", topo)
    rev_topo = topo
    rev_topo.reverse()
    print("REVERSE: ", rev_topo)

    #  c) schedule_functions(): walks topo_order backwards building the nests. `out` nest first, then the previous ones (next one in reversed order)
    #     in: env, topo_order, each Func schedule
    #     out: 1st Stmt tree, with For, Provide, Realize, ProducerConsumer nodes
    schedule_functions(ENV_TABLE, rev_topo, shape)

    #  d) bounds_inference():
    #     in: schedule_functions() resulting tree
    #     out: same tree with LetStmt nodes that give the symbols values
    #  e) (Skip for now): sliding_window, storage_folding, ...
    #  f) storage_flattening():
    #     in: bounds_inference() tree
    #     out: same tree but Realize becomes Allocate/Free, Provide becomes Store,
    #         Calls become Load, all with 1D indices (flattens)
    #  g) vectorize, unroll, simplify: For nodes tagged vectorize/unrolled get rewritten, and 
    #      consts get folded
    #  h) codegen:
    

    # c) schedule_functions


def realize(func, shape): 
    # What realize does:
    #   e.g. out.realize([4,8])
    #   1. allocates 4x8 buffer outside compiled code. that's why out never gets realized
    x,y = shape
    buf = [0.0] * (x*y)

    #   2. calls lower(). lower() calls:
    lower(func, shape)

"""
Schedule decision:
For each function we ask:
- is this Func formula (recompute every time someone asks) or is it a buffer in memory 
(compute it, store, then read it back)?

Case:
        f(i0,i1) = A(i0,i1) * 2
        g(i0,i1) = f(i0,i1) + 1
        out(i0,i1) = g(i0,i1) + f(i0,i1)

3 options:
    a) inline (formula) - No buffer, so everytime the function is mentioned, recomputes on the spot:
    Ex: if out is scheduled as inline, that means no buffer. So everytime 'f' is mentioned, recomputes
        on the spot.
        out would be:
        ```
        for (i1=0; i1<8; i1++)
            for (i0=0; i0<4; i0++)
                out[i0][i1] = (A[i0][i1]*2 + 1) + (A[i0][i1]*2); # this is out(i0,i1) = g(i0,i1)+f(i0,i1) 
        ```
        as we can see, 'f' and 'g' don't exist inside out inline, they're substituted on the spot.

    b) compute_root - buffer at the top. Give 'f' a real array, fill it first, then read it.
    Ex:
        ```
        int fbuf[4][8];                             # creates the buffer

        for (i1=0; i1<8; i1++)
            for (i0=0; i0<8; i0++)
                fbuf[i0][i1] = A[i0][i1]*2          # FILL THE BUFFER FIRST

        for (i1=0; i1<8; i1++)
            for (i0=0; i0<8; i0++)
                out[i0][i1] = (fbuf[i0][i1] + 1) + fbuf[i0][i1]; # this is the same out(i0,i1) = g(i0,i1) + f(i0,i1)
        ```

    c) compute_at - tiny buffer inside the loop that dies each iteration
        ```
        for (i1=0; i1<8; i1++)
            for (i0=0; i0<4; i0++)
                int fbuf[1];                            # creates a buffer with 1 element
                fbuf[0] = A[i0][i1] * 2                 # fills the buffer with 1 value
                out[i0][i1] = (fbuf[0] + 1) + fbuf[0]   # read the buffer
                                                        # buffer dies

    Translating 'b) compute_root' stage as the compiler IR, we have:
        int fbuf[4][8]          -> Realize f - the buffer scope starts
        for ... fbuf = A*2      -> Produce f - the loops that write the buffer
        for ... out = fbuf...   -> Consume f - the loops that read the buffer
        free(fbuf)              -> end of realize scope

    As a tree:
        Realize f:                #everything inside Realize f scope, means the buffer f is alive.
            Produce f:            # writer loops (Stores)
            Consume f:            # reader loops (Loads)
                
"""

