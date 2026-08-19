from typing import Union
from ir import Func, Input
from ops import op_add, op_sub, op_mul, op_div, op_sum, op_matmul, op_relu, op_exp, op_maxreduce, op_softmax

# wrapper
class Tensor:
    def __init__(self, func: Union[Func, Input], shape: tuple):
        # 'func' is a reference to either data-to-come(input) or a recipe(func)
        self.func = func # an array defined by a rule (Func) or an array arriving in runtime (Input). Also the 'ref'
        self.shape = shape

    @staticmethod
    def input(shape, dtype="int", name="in"): # this is how the user declares "data will arrive here at runtime"
        return Tensor(Input(name, dims=len(shape), type=dtype), shape)

    def __repr__(self) -> str:
        return f'Tensor({self.func!r}, {self.shape!r})'

    def __add__(self, other):
        return Tensor(*op_add(self, other))
        
    def __sub__(self, other):
        return Tensor(*op_sub(self, other))

    def __mul__(self, other):
        return Tensor(*op_mul(self, other))

    # maybe dont need here, same for op_sub
    def __div__(self, other):
        return Tensor(*op_div(self, other))

    def __matmul__(self, other):
        return Tensor(*op_matmul(self, other))
    
    def sum(self, axis=0):
        #return Tensor(*op_sum(self))
        return Tensor(*op_sum(self, axis))

    def relu(self):
        return Tensor(*op_relu(self))

    def exp(self):
        return Tensor(*op_exp(self))

    def maxreduce(self, axis=0):
        return Tensor(*op_maxreduce(self, axis))

    def softmax(self):
        return Tensor(*op_softmax(self))