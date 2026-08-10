from enum import Enum, auto
from typing import Union, Optional
from dataclasses import dataclass

# Hierarchy 1 - Expressions
class Expr:
    """ base class """
    pass 

@dataclass
class Call(Expr): # Call is "read one element of that thing (name), at these coordinates"
    name: str        # thing
    args: list[Expr] # coordinates
    ref: Union["Func","Input"] # ptr to the thing
    call_type: "CallType" #str # tag saying what kind of thing the "name" refers to.

    def __repr__(self) -> str:
        return f'Call({self.name}, {self.args}, {self.call_type})'

@dataclass
class Variable(Expr): # reference by name to a Var, RVar, or Let binding
    name: str

    def __repr__(self) -> str:
        return f'Variable({self.name})'

@dataclass
class Add(Expr):
    op0: Expr
    op1: Expr

    def __repr__(self) -> str:
        return f'Add({self.op0}, {self.op1})'

@dataclass
class Mul(Expr):
    op0: Expr
    op1: Expr

    def __repr__(self) -> str:
        return f'Mul({self.op0}, {self.op1})'

@dataclass
class ConstInt(Expr): # substitute IntImm for this one. more intuitive
    value: int

    def __repr__(self) -> str:
        return f'ConstInt({self.value})'

# hierarchy 2 - Structure
class Func: # represent a tensor or computation stage. has name, dims, 
    def __init__(self, name: str, args: list["Var"], value: Expr, updates: Optional[list[tuple]]):
        self.name = name
        self.args = args # indices that the computation will operate (i,j)
        self.value = value
        self.updates = updates # [(lhs_args, rhs, rdom)]
        #self.schedules = schedules

    def __repr__(self) -> str:
        return f'Func({self.name}, {self.args}, {self.value}, {self.updates})'

class Input: # external input, previously ImageParam from halide's
    def __init__(self, name, dims, type):
        self.name = name
        self.dims = dims
        self.type = type

    def __repr__(self) -> str:
        return f'Input({self.name}, {self.dims}, {self.type})'

class Var: # just a name
    def __init__(self, name):
        self.name = name
    
    def __repr__(self) -> str:
        return f'Var({self.name})'

class RDom:
    def __init__(self, name: str, min: Expr, extent: Expr): # I dont like extent. maybe change to 'range' or 'max'
        self.name = name
        self.min = min
        self.extent = extent

    def __repr__(self) -> str:
        return f'RDom({self.name}, {self.min}, {self.extent})'

# types
class CallType(Enum):
    INPUT = auto()
    FUNC = auto()

class DType(Enum):
    int32 = auto()
    float32 = auto()