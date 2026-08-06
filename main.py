from tracer import *

if __name__ == "__main__":
    a = Tensor.input((4,))
    b = Tensor.input((4,))
    c = a + b
    out = c.sum()
    print(out)

