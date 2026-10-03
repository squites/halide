#from tracer import topo
from ir import ForType
from typing import Union, Optional


class Schedule:
    def __init__(
        self, 
        dims: list[tuple(str, ForType)], # change to (str, ForType)
        splits: Optional[list["Split"]] = [], 
        compute_lvl: Union["LoopLevel", str] = "inline", # can be LoopLevel(fname, varname), or INLINE or ROOT
        store_lvl: Union["LoopLevel", str] = "inline" # str for "inline", "root". Change this later instead of str
    ) -> None:
        self.dims = dims
        self.splits = splits
        self.compute_lvl = compute_lvl
        self.store_lvl = store_lvl

    def __repr__(self) -> str:
        return f'Schedule({self.dims}, {self.splits}, compute={self.compute_lvl}, store={self.store_lvl})'
    
    # these should be inside Func() class?
    #def compute_at():
    #    pass
    #   
    #def split():
    #    pass

class Split(Schedule):
    def __init__(self, old, outer, inner, factor, tail) -> None:
        self.old_var = old
        self.outer_var = outer
        self.inner_var = inner
        self.factor = factor
        self.tail_strategy = tail

class LoopLevel(Schedule):
    def __init__(self, func_name: str, var_name: str) -> None:
        self.func_name = func_name
        self.var_name = var_name

    #@classmethod
    #def inlined(cls):
    #    return cls()
    ## def isinlined(): pass
    ## def isroot(): pass
    #@classmethod
    #def root(cls):
    #    return cls()
    

def schedule_functions():
    pass

def bounds_inference():
    pass

def storage_flattening():
    pass

"""
Explanation of schedules:
* compute_root(): means compute this func exactly once, over its entire needed domain, 
                  store it in a buffer, before any consumer runs
"""

