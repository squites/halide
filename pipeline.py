# this will have the realize function. So the builder will generate a Stmt tree. It'll receive as 
# input the env table + toposort list of nodes
from ops import find_transitive_calls, calldfs, ENV_TABLE
from ir import Func

def schedule_functions(env, rev_topo):
    print("\nSCHEDULE FUNCTIONS:")
    for i in rev_topo:
        print("**", env[i])

def bounds_inference():
    pass

def storage_flattening():
    pass

def realization_order(out, table):
    return calldfs(out, table)

def lower(out):
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
    schedule_functions(ENV_TABLE, rev_topo)
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
    lower(func)


