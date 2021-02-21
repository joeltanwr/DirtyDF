import pandas as pd
from numpy.random import default_rng
from time import time
from stainer_new import *
from history import *

"""
Edits to implement:
1. Column-types
    Allow column-type interpretation for DDF
    For each stainer, a (global) col_type attribute should be added.
    In the DDF class, before a stainer is ran, the col_type should be queried and the DDF should
        retrieve the relevant column types
    This should then be checked against the stainer columns
    Map the relevant columns then call the transform

2. Function that will handle the mapping (Some sort of function that will trace back the ordering)
    - DONE

3. reindex stainers
    Reorder the stainer list into another ordering

4. summarise_stainers

5. randomize stainer order
    Not too sure how important this is but may require some sanity checks if implemented

6. Documentation
"""

class DirtyDF:
    """
    row_map / col_map stores the initial --> current mapping ({original: end})
    """
    def __init__(self, df, seed = None, copy = False):
        self.df = df
        
        if not copy:
            if not seed:
                self.seed = int(time() * 100 % (2**32 - 1))
            else:
                self.seed = seed

            self.rng = default_rng(self.seed)
            self.stainers = []
            self.row_map = {i: i for i in range(df.shape[0])} 
            self.col_map = {i: i for i in range(df.shape[1])}
            self.history = [] 
    
    def get_df(self):
        return self.df
    
    def get_seed(self):
        return self.seed
    
    def get_rng(self):
        return self.rng

    def get_mapping(self, axis = 0):
        if axis in (0, "row"):
            return self.row_map
        if axis in (1, "column"):
            return self.col_map
        raise Exception("Invalid axis parameter")

    def get_map_from_history(self, index, axis = 0):
        if axis in (0, "row"):
            return self.history[index].get_row_map()
        if axis in (1, "col"):
            return self.history[index].get_col_map()
        raise Exception("Invalid axis parameter")
        
    def get_previous_map(self, axis = 0):
        return self.get_map_from_history(-1, axis)

    def reset_rng(self):
        self.rng = default_rng(self.seed)
    
    # Print methods
    def summarise_stainers(self):
        """ Will likely depend on some other method within Stainer """
        pass
    
    def __add_history__(self, message, row_map, col_map):
        self.history.append(History(message, row_map, col_map))
    
    def add_stainers(self, stainer, use_orig_row = True, use_orig_col = True):
        ddf = self.copy()
        if isinstance(stainer, Stainer):
            ddf.stainers.append((stainer, use_orig_row, use_orig_col))
        else:
            for st in stainer:
                ddf.stainers.append((st, use_orig_row, use_orig_col))
        
        return ddf
        
    def reindex_stainers(self, new_order):
        """
        Reorder stainers
        """
        pass
    
    def run_stainer(self, idx = 0):
        ddf = self.copy()
        stainer, use_orig_row, use_orig_col = ddf.stainers.pop(idx)
        
        row, col = stainer.get_indices()
        
        n_row, n_col = self.df.shape
        
        if not row:
            row = [i for i in range(n_row)]
        if not col:
            col = [i for i in range(n_col)]
        
        if not use_orig_row:
            row = list(map(lambda x: self.row_map[x], row))
        if not use_orig_col:
            col = list(map(lambda x: self.col_map[x], col))      
        
        res = stainer.transform(self.df, self.rng, row, col)
        
        try:
            new_df, row_map, col_map = res
        except:
            raise Exception("Need to enter a row_map and col_map. If no rows or columns were added/deleted, enter an empty dictionary")

        # Default options
        if not row_map:
            row_map = {i: i for i in range(new_df.shape[0])}
        if not col_map:
            col_map = {i: i for i in range(new_df.shape[1])}

        def new_mapping(original, new):
            """
            Given an old mapping and a one-step mapping, returns a mapping that connects the most
            original one to the final mapping 
            """
            o_dict = original.items()
            o_key = list(map(lambda x: x[0], o_dict))
            o_val = list(map(lambda x: x[1], o_dict))

            return dict(zip(o_key, map(lambda x: new[x], o_val)))

        ddf.row_map = new_mapping(self.row_map, row_map)
        ddf.col_map = new_mapping(self.col_map, col_map)
        
        ddf.__add_history__(stainer.get_history(), row_map, col_map) # This stores the -1 mapping
        ddf.df = new_df
        return ddf
    
    def run_all_stainers(self, rng = None):        
        current_ddf = self
        for stainer in self.stainers:
            current_ddf = current_ddf.run_stainer()
        return current_ddf

    def copy(self):
        new_ddf = DirtyDF(self.df.copy(), copy = True)
        new_ddf.seed = self.seed
        new_ddf.rng = self.rng
        new_ddf.stainers = self.stainers.copy()
        new_ddf.history = self.history.copy()
        new_ddf.row_map = self.row_map.copy()
        new_ddf.col_map = self.col_map.copy()
        return new_ddf
