from tracer import *
from matcher import *

if __name__ == "__main__":
    a = Node(op="INPUT", shape=(3,2), dtype="float32", name="A")
    b = Node(op="INPUT", shape=(2,3), dtype="float32", name="B")
    c = (a @ b)
    e = c.relu() # separate here to visualize better

    x = Node(op="INPUT", shape=(3,2), dtype="float32", name="X")
    y = Node(op="INPUT", shape=(2,3), dtype="float32", name="Y")
    d = (x @ y)
    f = d.relu()

    g = (e @ f)
    out = g.relu()

    print("Walking:")
    topo = [] # this is a list of topological sorted nodes
    walk(out, topo)
    print("\nTOPO: ")
    for i in topo:
        print("*", i.name)

    rFusion1 = Rule("fused_matmul_relu", pat_fused_matmul_relu, build_fused_matmul_relu)
    rules = [rFusion1]
    rewriter = Rewriter(rules, graph=out)
    y = rewriter.rewrite(topo, out)

    newtopo = []
    print("\nRERUNNING TOPO:")
    walk(y, newtopo)
    for i in newtopo:
        print("*", i.name)

    
#test_walk(topo)
#graph = viz_graph(d)
#graph.save("graph.dot")
#graph.render("graph.dot", view=True)
