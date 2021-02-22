from time import time
import numpy as np
import pandas as pd

"""
To-do:
1. Fix initialisation of history so that it would reset upon creation of DDF
    - DONE

2A. AddDuplicate Stainer
2B. Nullify Stainer
2C. Function Stainer

3. Currently updating history is only message. Will need to adjust more if
want it to contain other information
    - Included time
"""

class Stainer:
    col_type = "all"
    
    def __init__(self, name = "Unnamed Stainer", row_idx = [], col_idx = []):
        self.name = name
        self.row_idx = row_idx
        self.col_idx = col_idx
        
        self.__initialize_history__()

    def get_col_type(self):
        return self.col_type

    def get_indices(self):
        return self.row_idx, self.col_idx
    
    def transform(self, df, rng):
        raise Exception("Stainer not implemented")

    def init_transform(self, df, row, col):
        """
        Helper method to assign df / row / cols before transforming
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df should be pandas DataFrame')
        
        new_df = df.copy()
        
        if not row:
            row = self.row_idx
        if not col:
            col = self.col_idx

        return new_df, row, col
                           
    # History-related methods #
    def __initialize_history__(self):
        self.message = ""
        self.time = 0
    
    def update_history(self, message = "", time = 0):
        self.message += message
        self.time += time
    
    def get_history(self):
        """ Creates a history object and returns it"""
        msg, time = self.message, self.time
        if not time:
            time = "Time not updated. Use update_history to update time"
        self.__initialize_history__()
        return self.name, msg, time

class ShuffleStainer(Stainer):
    """ This description isn't complete """ 
    
    def __init__(self, name = "Shuffle"):
        super().__init__(name, [], [])
        
    def transform(self, df, rng, row = None, col = None):
        new_df, row, col = self.init_transform(df, row, col)

        start = time()
        # Shuffle + Create mapping
        new_df = new_df.sample(frac = 1, random_state = rng.bit_generator)
        original_idx = new_df.index
        new_df.reset_index(drop = True, inplace = True)
        new_idx = new_df.index

        row_map = np.zeros((df.shape[0], df.shape[0]))
        for i in range(len(row_map)):
            row_map[original_idx[i]][new_idx[i]] = 1

        end = time()
        
        self.update_history("Order of rows randomized", end - start)
        
        return new_df, row_map, {}        
