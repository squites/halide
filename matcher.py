from tracer import Node

class Rule:
    def __init__(self, name, pattern, builder):
        self.name = name
        self.pattern = pattern
        self.builder = builder

    def match(self, node, graph):
        fused = self.pattern(node) # if pattern returns the binds, means match
        if fused is not None:
            print("MATCH!")
            return fused
        print("NO MATCH!")
        return None

class Rewriter:
    def __init__(self, rules, graph):
        self.rules = rules
        self.graph = graph # remove later

    def rewrite(self, topo):
        # apply each rule to each node of subgraph
        for node in topo:
            for rule in self.rules:
                print(f'\nMatching rule:{rule.name}, node:{node}')
                fused = rule.match(node, topo)
                if fused is not None:
                    newnode = rule.builder(fused)
                    print("\nOLD: ", node)
                    print("\nNEW: ", newnode)
                    for old in topo: # self
                        for i, input in enumerate(old.inputs):
                            if input is node:
                                old.inputs[i] = newnode

def pat_fused_matmul_relu(subgraph): # subgraph is also a node (ex: Node RELU is a subgraph)
    print("\nPATTERN\nnode: ", subgraph)
    binds = {}
    if subgraph.op == "RELU" and subgraph.inputs[0].op == "MATMUL":
        print("FUSE!")
        binds["inputs"] = [f for f in subgraph.inputs[0].inputs]
        binds["shape"] = subgraph.shape
        binds["dtype"] = subgraph.dtype
        return binds
    return None

 
def build_fused_matmul_relu(binds):
    print(f'\nBINDS:\n{binds["inputs"]}\n{binds["shape"]}\n{binds["dtype"]}')
    name = f'{binds["inputs"][0].name}{binds["inputs"][1].name}' 
    node = Node(op="F_MATMUL_RELU", inputs=binds["inputs"], shape=binds["shape"], dtype=binds["dtype"], name=f"FUSED_MATMUL_RELU({name})")
    print("\nBUILD\n", node)
    return node

# BUG: matmul of fused nodes as input are not working, because the matmul node still kept the old
# input nodes "RELU" instead of the new ones "FUSED"

"""
Tinygrad PatternMatcher:

pm = PatternMatcher([
  #  (x, y): if finds pattern `x`, replace with `y`
  (UPat(Ops.CONST, arg=0), lambda: UOp.const(dtypes.int, 42)),
])
zero = UOp.const(dtypes.int, 0)
print(pm.rewrite(zero)) # UOp(Ops.CONST, dtypes.int, arg=42, src=())

ex2:

pm = PatternMatcher([
    # ADD(x,x) -> x * 2
    (UPat(ADD, src=(x, x)), lambda x: x * 2)
])
"""