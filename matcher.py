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

    def rewrite(self, topo, root):
        # apply each rule to each node of subgraph
        newroot = root
        for node in topo:
            for rule in self.rules:
                print("\nNODE: ", node.name)
                x = rule.match(node, topo)
                if x is not None:
                    fused = rule.builder(x, node.name)
                    # We fuse the node but we need to rewire the inputs. Basically, if the default
                    # node that we fuse is input for any other node in the graph, then we need to 
                    # rewire, so instead of the default node, the input will be the new fused one 
                    for other in topo:
                        for i, input in enumerate(other.inputs):
                            if input is node:
                                #print("EQUAL!")
                                other.inputs[i] = fused
            if node == newroot and fused is not None:
                newroot = fused
        return newroot

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

def build_fused_matmul_relu(binds, name):
    print(f'\nBINDS:\n{binds["inputs"]}\n{binds["shape"]}\n{binds["dtype"]}')
    node = Node(op="F_MATMUL_RELU", inputs=binds["inputs"], shape=binds["shape"], dtype=binds["dtype"], name=f"FUSED_MATMUL_RELU({name})")
    print("\nBUILD\n", node)
    return node

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