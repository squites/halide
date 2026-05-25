from typing import Optional, Union
from dataclasses import dataclass, field
import graphviz
#@dataclass
#class Tensor:
#    node: "Node"
#
#    def __matmul__(self, other):
#        return Tensor(Node(op="MATMUL", inputs=[self.node, other.node]))

@dataclass
class Node:
    op: str# = "data"
    inputs: list["Node"] = field(default_factory=list)
    shape: Optional[tuple] = None
    dtype: Optional[str] = None #Union[int, float] # str for now. Later create a class Dtype()
    name: Optional[str] = None

    def __matmul__(self, other):
        M, K1 = self.shape
        K2, N = other.shape
        if K1 != K2:
            raise RuntimeError(f"Inner dimension mismatch: {self.shape} x {other.shape}")
        return Node(op="MATMUL", inputs=[self,other], shape=(M, N), name="c")

    def relu(self):
        return Node(op="RELU", inputs=[self], shape=self.shape, name="d")


def walk(node, topo, visited=None):
    """ walk the graph """
    print("* node:", node.name)
    print("visited (BEFORE): ", visited)
    
    if visited is None:
        visited = set()
    
    if node.name in visited:
        return node
    visited.add(node.name)
    print("visited (AFTER): ", visited)
   
    #if node.inputs:
    for i in node.inputs:
        print(f"input node {node.name}: ", i)
        walk(i, topo, visited)
    topo.append(node.name)
    #print("TOPO: ", topo)

    return node

def viz_graph(root, dot=None, visited=None):
    if dot == None:
        dot = graphviz.Digraph("Computation Graph", format="png")
        dot.attr(rankdir="BT")
    
    if visited is None:
        visited = set()

    #if root in visited:
    #    return dot
    visited.add(root.name)

    label = f'{root.name}'
    label += f'op:{root.op}'
    if root.shape:
        label += f'shape:{root.shape}'

    dot.node(name=root.name, label=label, shape="box")
    if root.inputs:
        for i in root.inputs:
            viz_graph(i, dot, visited)
            dot.edge(i.name, root.name)
    
    return dot

# oracle? remove this later
def check_matmul(a, b, c, M, N, K):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                c[i][j] += a[i][k] * b[k][j]

a = Node(op="INPUT", shape=(3,2), dtype="float32", name="A")
b = Node(op="INPUT", shape=(2,3), dtype="float32", name="B")
c = (a @ b)
d = c.relu() # separate here to visualize better

print("\nMatmul:")
print(c)

print("\nRelu:")
print(d, "\n")

print("Walking:")
topo = [] # this is a list of topological sorted nodes
walk(d, topo)
print("TOPO: ", topo)

#graph = viz_graph(d)
#graph.save("graph.dot")
#graph.render("graph.dot", view=True)

# TODO: add separate pass that walks on the full graph in topo order, and fills in the shapes. 


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
"""