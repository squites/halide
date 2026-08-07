from typing import Union
from ir import Func, Input
from ops import op_add, op_mul, op_sum

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

    def __mul__(self, other):
        return Tensor(*op_mul(self, other))
    
    def sum(self):
        return Tensor(*op_sum(self))