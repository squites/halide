from enum import Enum, StrEnum, auto
from typing import Union, Optional
from dataclasses import dataclass

class Expr:
    """ base class """

    def children(self):
        return [] 

@dataclass
class Call(Expr): # Call is "read one element of that thing (name), at these coordinates"
    name: str        # thing
    args: list[Expr] # coordinates
    ref: Union["Func","Input"] # ptr to the thing
    call_type: "CallType" # tag saying what kind of thing the "name" refers to.

    def __repr__(self) -> str:
        return f'Call({self.name}, {self.args}, ref={self.ref.name}, {self.call_type})'
    
    def children(self):
        return self.args

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

    def children(self):
        return [self.op0, self.op1]

@dataclass
class Sub(Expr):
    op0: Expr
    op1: Expr
    
    def __repr__(self) -> str:
        return f'Sub({self.op0}, {self.op1})'
    
    def children(self):
        return [self.op0, self.op1]

@dataclass
class Mul(Expr):
    op0: Expr
    op1: Expr

    def __repr__(self) -> str:
        return f'Mul({self.op0}, {self.op1})'

    def children(self):
        return [self.op0, self.op1]

@dataclass
class Div(Expr):
    op0: Expr
    op1: Expr

    def __repr__(self) -> str:
        return f'Div({self.op0}, {self.op1})'

    def children(self):
        return [self.op0, self.op1]

@dataclass
class Max(Expr):
    op0: Expr
    op1: Expr

    def __repr__(self) -> str:
        return f'Max({self.op0}, {self.op1})'

    def children(self):
        return [self.op0, self.op1]

@dataclass
class ConstInt(Expr): # substitute for IntImm
    value: int

    def __repr__(self) -> str:
        return f'ConstInt({self.value})'

@dataclass
class ConstFloat(Expr):
    value: float

    def __repr__(self) -> str:
        return f'ConstFloat({self.value})'

# --- 
class Func:
    def __init__(self, name: str, args: list["Var"], value: Expr, updates: Optional[list[tuple]], schedules: Optional[list]):
        self.name = name
        self.args = args # indices that the computation will operate (i,j)
        self.value = value
        self.updates = updates # [(lhs_args, rhs, rdom)]
        self.schedules = schedules # should be Schedule object?
        # default schedule: "compute_root". halide's default is actually "inline". implement later
        # when lowering this will write a simple loop per dimension nested + a full size allocation for the Func
        # ex: add2(i0, i1). this will generate:
        # alloc add2[8][32]
        # for i0 in 0..8:
        #   for i1 in 0..32:
        #       ...

    def __repr__(self) -> str:
        return f'Func({self.name}, {self.args}, {self.value}, {self.updates}, {self.schedules})'

    def children(self):
        return [self.args, self.updates]

    def realize(self, shape):
        import pipeline as pipe
        return pipe.realize(self, shape)
        
    
    # schedule record functions
    def compute_root(self):
        self.schedules.compute_lvl = "root"
        self.schedules.store_lvl = "root"

    def inline(self):
        self.schedules.compute_lvl = "inline"
        self.schedules.store_lvl = "inline"

        

class Input: # external input, previously ImageParam from halide's
    def __init__(self, name, dims, type):
        self.name = name
        self.dims = dims
        self.type = type

    def __repr__(self) -> str:
        return f'Input({self.name}, {self.dims}, {self.type})'

class Var:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self) -> str:
        return f'Var({self.name})'

class RDom:
    def __init__(self, name: str, min: Expr, extent: Expr): # change to 'range' or 'max'
        self.name = name
        self.min = min
        self.extent = extent

    def __repr__(self) -> str:
        return f'RDom({self.name}, {self.min}, {self.extent})'

# --- 
# OBS: Stmt has 2 IRs:
# 1) Early Lowering (Realize, Produce, Consume, Store). Func-level terms
# 2) Late Lowering (Allocate, For, Load/Store, Block, ...). memory-level terms
# This will be the same IR. A series of passes will rewrite in a way that the IR 
# will transform 
class Stmt:
    pass

"""
Realize():
 Allocate a multi-dimensional buffer of the given type and
 * size. Create some scratch memory that will back the function 'name'
 * over the range specified in 'bounds'. The bounds are a vector of
 * (min, extent) pairs for each dimension. Allocation only occurs if
 * the condition evaluates to true.
"""
class Realize(Stmt):
    def __init__(self, func: Func, bound: tuple, body: Stmt, dtype: DType):
       self.func = func
       self.bound = bound
       self.body = body
       self.dtype = dtype

    def __repr__(self):
        return f'Realize({self.func}, {self.bound}, {self.body}, {self.dtype})'

class Produce(Stmt):
    def __init__(self, name, body):
        self.name = name
        self.body = body

    def __repr__(self):
        return f'Produce({self.name}, {self.body})'
        
class Consume(Stmt):
    def __init__(self, name, body):
        self.name = name
        self.body = body

    def __repr__(self):
        return f'Consume({self.name}, {self.body})'

class Provide(Stmt):
    def __init__(self, name, args, value):
        self.name = name
        self.args = args
        self.value = value

    def __repr__(self):
        return f'Provide({self.name}, {self.args}, {self.value})'

class For(Stmt):
    def __init__(self, name: str, min: Expr, extent: Expr, fortype: ForType, body: Stmt):
        self.name = name
        self.min = min
        self.extent = extent
        self.fortype = fortype
        #self.device_api = device_api
        self.body = body

    def __repr__(self):
        return f'For({self.name}, min={self.min}, extent={self.extent}, {self.fortype}, {self.body})'

class Allocate(Stmt):
    pass

# types (maybe move types to other file)
class CallType(Enum):
    INPUT = auto()
    FUNC = auto()
    EXTERN = auto()

class DType(Enum):
    int32 = auto()
    float32 = auto()

# Add the ForType here?
class ForType(StrEnum):
    SERIAL = "serial"
    PARALLEL = "parallel" 
    VECTORIZED = "vectorized"
    UNROLLED = "unrolled"
    GPU_BLOCK = "gpu-block"
    GPU_THREAD = "gpu-thread"

"""
SCHEDULE:
# 1) each Func owns Schedule object. it has:
    - dims: ordered list {varname, fortype}. 
            - fortype can be: serial/parallel/vectorized/unrolled/gpu-block/gpu-thread 
            - the ordered list defines the loop nesting order and execution kind
    - splits: ordered list of {oldvar, outervar, innervar, factor, tail_strat}.
            - Order matters: ex: split(x, xo, xi, 8) then split(xo, xoo, xoi, 4). 
            - This will be applied in the order it was written
    - compute_level: LoopLevel = (func_name, var_name) pair + 2 sentinel values: inline(default) and root
    - store_level: LoopLevel where the allocation goes. inline(default)
Ex:
    * `g.compute_at(f, i0)`, we're doing g.schedule.compute_level = ("f", "i0")
    * `g.split(x, xo, xi, 8)`, we're doing g.schedule.splits.append(x, xo, xi, 8). So,
      we're appending this split rule to the splits list.
    * `g.reorder(y, x)`: reorder g's dims list
    * `g.parallel(x)`: set the fortype field on x's dims entry
    * `g.compute_at(f, i)`: set g's compute_lvl

After these calls, we have 2 things:
    - Graph of Funcs
    - Schedule records attached to each Func

    These are the 2 inputs to lowering!

IMPORTANT: NOTHING compiles. NOTHING transforms. IR is not touched. All compilation happens at
            realize() / compile_jit(), which read these records

# 2) Schedule fields separate into 2 families:
    a) nest shape: (read while constructing Func's loop nest): dims, splits, fortypes. 
        - Determine what the nest looks like
    
    b) placement: (read to decide where the finished nest goes in the final tree): compute_level, store_level


# 3) How to build one loop nest (nest shape):


** BIG PICTURE: **
Two phases:
1) User write the schedules by calling the directives. Each directive only 
   mutates Schedule fields:
    - g.split(x, xo, xi, 8) -> append {old: x, outer: xo, inner: xi, factor: 8} to
        g's splits list
    - g.reorder(y, x) -> reorder g's dims list
    - g.parallel(x) -> set the fortype field x's dims entry
    - g.compute_at(f, i) -> set g's compute level

    after all this, nothing is computed, only manipulate the Schedule obj of the Func g.

2) realize() triggers lowering (Schedule records get read).
    - builder() comes in. builder is the code that constructs the loop nests
    - the idea of the function is to turn "Func graph + Schedule records" into
      one big Stmt tree.  

TODO: 
- implement a schedule record + realize() and get a runnable compiler. 1st "inline"
- also build a validator, that validades the schedule records. So this doesnt happen: 
    * parallelizing reduction var: creates data race, because multiple threads accumulate into the
      same var concurrently, giving wrong results. Validator must forbid parallel/vectorize RDom vars
    * bad compute_at targets: ex: g.compute_at(f,v) where f doesn't consume g or v isn't var of f.
    * degenerate directives: split factor <= 0 , reorder duplicates, nonexistent var names
    * store_at/compute_at inconsistency. Can't allocate a buffer deeper than where it's filled

- pure def = func.value -> pure stage
- update def = func.updates -> update stage
pure stage gets loops over i0. update stage gets loops over i0 and r.
ex:  - "Split the update stage's i0 loop by 4" — affects only the update nest.                                         
     - "Reorder r outside i0" — only meaningful for the update stage; the pure stage doesn't even have an r loop.      
     - "Vectorize the pure init's i0" — affects only the init nest

OBS: move dims/splits/fortypes into per-stage schedules (list   
     of Schedule objects indexed by stage), keep compute/store level on the Func. Do it when you find yourself         
     wanting to split or reorder a reduction's r loop differently from its init

Resumo:    
    Here is what i understand so far. The first thing is to write the schedules. 
    So when I write the function g.compute_at(..) or f.split(..), this function will record
    the schedule inside the Func that called the function. 
    Example: g was Func(g,..., Schedule(dims=[x,y]...)) then  
    g.reorder(y,x) -> Func(g, ..., schedules=Schedule(dims=[y,x],...)) 
    This is what will happen. So nothing is computed yet, only records and manipulate the schedule 
    inside the Func object. Then when we call realize() the builder will construct the loops nests 
    transforming the Func graph with Schedule records into a Stmt tree. 

"""