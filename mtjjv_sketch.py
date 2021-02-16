import pandas as pd
from numpy.random import default_rng
from time import time
from stainer_new import *

def get_relevant_col_types(df, dtype):
    pass

class DirtyDF:
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
    
    def reset_rng(self):
        self.rng = default_rng(self.seed)
    
    # Print methods
    def summarise_stainers(self):
        """ Will likely depend on some other method within Stainer """
        pass
    
    def __add_history__(self, hist):
        if isinstance(hist, History):
            self.history.append(hist)
        else:
            raise TypeError('hist should be History object')
    
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
        
        ddf.__add_history__(stainer.get_history())
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
