import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_categorical_dtype
from numpy.random import default_rng
from time import time
from functools import reduce
from warnings import warn
from stainer import *
from history import *

class DirtyDF:
    """
    row_map / col_map stores the initial -> current mapping (row are the old, cols are new)
        1 represents a map; 0 o/w
    """
    def __init__(self, df, seed = None, copy = False):
        self.df = df
        
        if not copy:
            if not seed:
                self.seed = int(time() * 100 % (2**32 - 1))
            else:
                self.seed = seed
            
            self.rng = default_rng(self.seed)
            self.orig_shape = df.shape
            self.stainers = []
            self.row_map = {i: [i] for i in range(df.shape[0])} 
            self.col_map = {i: [i] for i in range(df.shape[1])} 
            self.history = [] 
        
        self.cat_cols = [i for i in range(df.shape[1]) if is_categorical_dtype(df.iloc[:, i])]
        self.num_cols = [i for i in range(df.shape[1]) if is_numeric_dtype(df.iloc[:, i])]
        self.dt_cols = [i for i in range(df.shape[1]) if is_datetime64_any_dtype(df.iloc[:, i])]
    
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
        for i, stainer in enumerate(self.stainers):
            print(f"{i+1}. {stainer[0].name}")

    def print_history(self):
        tuple(map(lambda x: print(self.history.index(x) + 1, x, sep = ". "), self.history))
    
    def __add_history__(self, message, row_map, col_map):
        self.history.append(History(message, row_map, col_map))
    
    def add_stainers(self, stain, use_orig_row = True, use_orig_col = True):
        ddf = self.copy()
        if isinstance(stain, Stainer):
            ddf.stainers.append((stain, use_orig_row, use_orig_col))
        else:
            for st in stain:
                ddf.stainers.append((st, use_orig_row, use_orig_col))
        
        return ddf
        
    def reindex_stainers(self, new_order):
        """
        Reorder stainers
        """
        ddf = self.copy()
        ddf.stainers = list(map(lambda x: ddf.stainers[x], new_order))
        
        return ddf
    
    def shuffle_stainers(self):
        n = len(self.stainers)
        new_order = self.rng.choice([i for i in range(n)], size = n, replace = False)
        return self.reindex_stainers(new_order)
    
    def run_stainer(self, idx = 0):
        ddf = self.copy()
        stainer, use_orig_row, use_orig_col = ddf.stainers.pop(idx)
        
        row, col = stainer.get_indices()
        
        n_row, n_col = self.orig_shape
        
        default_given = False
        if not row:
            row = [i for i in range(n_row)]
        if not col:
            col = [i for i in range(n_col)]
            default_given = True
        
        if use_orig_row:
            final_row = []
            for ele in row:
                final_row.extend(self.row_map[ele])
            row = final_row

        if use_orig_col:
            final_col = []
            for ele in col:
                final_col.extend(self.col_map[ele])
            col = final_col
        
        col_type = stainer.get_col_type()
        if col_type == "all":
            col = col
        elif col_type not in ("category", "cat", 
                              "datetime", "date", "time", 
                              "numeric", "int", "float"):
            warn(f"Invalid Stainer Column type for {stainer.name}. Using all columns instead")
        else:
            input_cols = set(col)
            if col_type in ("category", "cat"):
                relevant_cols = set(ddf.cat_cols)
            if col_type in ("datetime", "date", "time"):
                relevant_cols = set(ddf.dt_cols)
            if col_type in ("numeric", "int", "float"):
                relevant_cols = set(ddf.num_cols)
            if not default_given and not input_cols.issubset(relevant_cols):
                raise TypeError(f"Column with incorrect column type provided to stainer {stainer.name}, which requires column type {col_type}.")
            else:
                col = list(input_cols & relevant_cols)
        
        res = stainer.transform(self.df, self.rng, row, col)
        
        try:
            new_df, row_map, col_map = res
        except:
            raise Exception("Need to enter a row_map and col_map. If no rows or columns were added/deleted, enter an empty list")

        # Default options
        if not len(row_map):
            row_map = {i: [i] for i in range(new_df.shape[0])} 
        if not len(col_map):
            col_map = {i: [i] for i in range(new_df.shape[1])} 
        
        def new_mapping(original, new):
            """
            Given an old mapping and a one-step mapping, returns a mapping that connects the most
            original one to the final mapping 
            """
            final_map = {}
            for k, v in original.items():
                final_map[k] = []
                for element in v:
                    final_map[k].extend(new[element])
            """
            final_map = np.zeros((len(original), len(new[0])))
            for i in range(len(original)):
                initial_map = np.nonzero(original[i])[0]
                new_idx = reduce(lambda x, y: np.concatenate([x,y]).reshape(-1),
                             map(lambda x: np.nonzero(new[x])[0], initial_map))
                if len(new_idx):
                    final_map[i][new_idx] = 1
            return final_map
            """
            return final_map


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
        new_ddf.orig_shape = self.orig_shape
        new_ddf.stainers = self.stainers.copy()
        new_ddf.history = self.history.copy()
        new_ddf.row_map = self.row_map.copy()
        new_ddf.col_map = self.col_map.copy()
        return new_ddf