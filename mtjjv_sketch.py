import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_categorical_dtype
from numpy.random import default_rng

"""
Difficulty:
Stainer class requires user to fill in the map

Need to confirm how the RNG object works
"""

class DirtyDF:
    """
    Available attributes:
    df
    orig_df         Original dataframe
    stainers        List of stainers
    history         List of History objects (past actions)
    col_names_dict  col number -> col name (updated to newest df)
    row_map         old row numbers -> new row numbers
    col_map         old col numbers -> new col numbers
    cat_cols        List of categorical columns (names)
    num_cols        List of numerical columns (names)
    dt_cols         List of datetime columns (names)

    Available methods:
    get_df
    describe_stainers
    addStainer
    
    dirty
    copy
    """
    # Store DF
    # Store stainer list
    # Store history
    def __init__(self, df, rng = None, cat_cols = []):
        if isinstance(df, pd.DataFrame):
            self.df = df
            self.orig_df = df
        else:
            raise TypeError('df should be pandas DataFrame')

        if rng:
            self.rng = rng
        else:
            self.rng = default_rng(int(time() * 100 % (2**32 - 1)))
            
        self.stainers = []
        self.history = []
        self.col_names_dict = dict(enumerate(df.columns))
        
        self.row_map = {i: i for i in range(df.shape[0])}
        self.col_map = {i: i for i in range(df.shape[0])}

        self.cat_cols = list(map(lambda x: col_map[x] if type(x) == int else x, cat_cols))
        self.cat_cols.extend(list(filter(lambda x: is_categorical_dtype(df[x]), df.columns)))
        self.num_cols = [col for col in df.columns if is_numeric_dtype(df[col]) and col not in cat_cols]
        self.dt_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col]) and col not in cat_cols]

    # Getters 
    def get_df(self):
        return self.df

    def get_rng(self):
        return self.rng

    def get_col_name(self, orig_idx = None, curr_idx = None):
        if orig_idx == None and curr_idx == None:
            raise ValueError("Need to provide index number of column to index")
        elif orig_idx != None and curr_idx != None:
            raise ValueError("Only should provide one argument")
        else:
            idx = orig_idx if orig_idx != None else curr_idx
            try:
                if orig_idx != None:
                    return self.col_names_dict[self.col_map[orig_idx]]
                return self.col_names_dict[curr_idx]
            except:
                raise TypeError("Index should be an integer")

        
    def categorical_cols(self):
        return self.cat_cols

    def numerical_cols(self):
        return self.num_cols

    def date_cols(self):
        return self.dt_cols

    # Print methods
    def summarise_stainers(self):
        pass
    
    # Setters
    def __add_history__(self, hist):
        if isinstance(df, History):
            self.history.append(hist)
        else:
            raise TypeError('hist should be History object')

    def add_stainer(self, stainer):
        if isinstance(df, Stainer):
            self.stainers.append(stainer)
        else:
            raise TypeError('stainer should be Stainer object')

    def reindex_stainers(self, new_order):
        """
        Reorder stainers
        """
        pass
    
    # Methods
    def run_stainer(self, idx = 0):
        stainer = self.stainers[idx]
        new_df, history = stainer.transform(self)
        new_ddf = self.copy()

        new_ddf.df = new_df.copy()
        new_ddf.__add_history__(history)
        new_ddf.stainers.pop(0)
        return new_ddf
                

    def run_all_stainers(self):
        current_ddf = self.df
        for stainer in self.stainers:
            current_ddf = self.run_stainer(current_ddf)
        return current_ddf

    # Copy
    def copy(self):
        new_ddf = DirtyDF(self.df.copy(), rng = self.rng, cat_cols = self.cat_cols)
        new_ddf.stainers = self.stainers.copy()
        new_ddf.history = self.stainers.copy()
        new_ddf.row_map = self.row_map.copy()
        new_ddf.col_map = self.col_map.copy()
        return new_ddf
        
    

class History:
    """
    How to format this?
    Should probably contain action + new mapping 
    """
    pass

        

"""
class Stainer


-- transform
takes in some DDF
Retrives the DF
does manipulation

returns a DF and History object
"""
