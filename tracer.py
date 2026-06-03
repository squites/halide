from typing import Optional
from dataclasses import dataclass, field
import graphviz

#from matcher import Rule, Rewriter

# Graph IR. Tensor-op level
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

        if self.dtype != other.dtype:
            raise RuntimeError(f"Data type mismatch: {self.dtype}, {other.dtype}")
        return Node(op="MATMUL", inputs=[self,other], shape=(M, N), dtype=self.dtype, name=f'MAT({self.name}{other.name})')

    def relu(self):
        return Node(op="RELU", inputs=[self], shape=self.shape, dtype=self.dtype, name=f'RELU_{self.name}')


def walk(node, topo, visited=None):
    """ walk the graph """
    if visited is None:
        visited = set()

    if node.name in visited:
        return node
    visited.add(node.name)

    for i in node.inputs:
        walk(i, topo, visited)
    topo.append(node)
    return node

def viz_graph(root, dot=None, visited=None):
    if dot == None:
        dot = graphviz.Digraph("Computation Graph", format="png")
        dot.attr(rankdir="BT")
    
    if visited is None:
        visited = set()
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
            tmp = 0
            for k in range(K):
                tmp += a[i][k] * b[k][j]
            c[i][j] = tmp

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
"""