import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_categorical_dtype
from numpy.random import default_rng
from time import time
from functools import reduce
from warnings import warn
from .stainer import Stainer
from .history import *

class DirtyDF:
    """
    Dirty DataFrame. Stores information about the dataframe to be stained, previous staining results, 
    and the mapping of the rows and columns.
    
    To be used in conjunction with Stainer class to add and execute stainers.
    """
    def __init__(self, df, seed = None, copy = False):
        """ 
        Constructor for DirtyDF
        
        Parameters
        ----------
        df : pd.DataFrame 
            Dataframe to be transformed.
        seed : int, optional
            Controls the randomness of the staining process. For a
            deterministic behaviour, seed has to be fixed to an integer. If unspecified, will choose a random seed
        copy : boolean, optional
            Not for use by user. Determines if a copy of DirtyDF is being
            created. If True, will copy the details from the previous DDF.
        """
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
        """ Returns the dataframe 
        
        Returns
        ----------
        df : pd.DataFrame
            Current dataframe in DDF
        """
        return self.df
    
    def get_seed(self):
        """ Returns seed number 
        
        Returns
        ----------
        seed : int
            Integer seed used to create Generator for randomisation 
        """
        return self.seed
    
    def get_rng(self):
        """ Returns seed generator
        
        Returns
        ----------
        rng : np.random.BitGenerator
            PCG64 pseudo-random number generator used for randomisation
        """
        return self.rng

    def get_mapping(self, axis = 0):
        """ Mapping of rows/cols from original dataframe to most recent dataframe. 
        A dictionary is returned with information on which index the original
        rows/cols are displayed in the newest dataframe. 
        For instance, if row 3 got shuffled to row 8 in the new dataframe, then
        row 8 got shuffled to row 2, the function will return {3: [2]}
        
        Parameters
        ----------
        axis : (0/1), optional
            If 0, returns the row mapping.
            If 1, returns the col mapping.
            
            Defaults to 0
        
        Returns
        ----------
        map : {int : int list} dictionary 
            Mapping of original row/col indices to current dataframe's row/col indices.
            
        Raises
        ----------
        Exception
            If axis provided is not 0/1
        """
        if axis in (0, "row"):
            return self.row_map
        if axis in (1, "column"):
            return self.col_map
        raise Exception("Invalid axis parameter")

    def get_map_from_history(self, index, axis = 0):
        """ Mapping of rows/cols of the sepcified stainer transformation that had been executed.
        A dictionary is returned with information on what row/col index right
        before the specified transformation has converted to after the
        transformation. 
        For instance, if row 3 got shuffled to row 8 in the new dataframe, then
        row 8 got shuffled to row 2, calling index=0 will return {3: [8]} 
        and calling index=1 will return {8: [2]}
        
        Parameters
        ----------
        index : int
            Index of stainer sequence to query mapping. E.g. index=1 will query
            the mapping performed by the 2nd stainer operation.
        axis : (0/1), optional
            If 0, returns the row mapping.
            If 1, returns the col mapping.
            
            Defaults to 0
        
        Returns
        ----------
        map : {int : int list} dictionary 
            Mapping of original row/col indices to current dataframe's row/col indices.
            
        Raises
        ----------
        Exception
            If axis provided is not 0/1
        """
        if axis in (0, "row"):
            return self.history[index].get_row_map()
        if axis in (1, "col"):
            return self.history[index].get_col_map()
        raise Exception("Invalid axis parameter")
        
    def get_previous_map(self, axis = 0):
        """ Mapping of rows/cols of the most recent stainer transformation that had been executed.
        A dictionary is returned with information on what row/col index right
        before the transformation has converted to after the transformation. 
        For instance, if row 3 got shuffled to row 8 in the new dataframe, then
        row 8 got shuffled to row 2, the function will return {8: [2]}
        
        Parameters
        ----------
        axis : (0/1), optional
            If 0, returns the row mapping.
            If 1, returns the col mapping.
            
            Defaults to 0
        
        Returns
        ----------
        map : {int : int list} dictionary 
            Mapping of original row/col indices to current dataframe's row/col indices.
            
        Raises
        ----------
        Exception
            If axis provided is not 0/1
        """
        return self.get_map_from_history(-1, axis)

    def reset_rng(self):
        """ Resets Random Generator object """
        self.rng = default_rng(self.seed)
    
    # Print methods
    def summarise_stainers(self):
        """ Prints names of stainers that have yet to be executed """
        for i, stainer in enumerate(self.stainers):
            print(f"{i+1}. {stainer[0].name}")

    def print_history(self):
        """ Print historical details of the stainers that have been executed """
        tuple(map(lambda x: print(self.history.index(x) + 1, x, sep = ". "), self.history))
    
    def __add_history__(self, data, row_map, col_map):
        """ Not to be explicitly called by user. Used in conjunction while running stainer to create and add History object to DDF information.
        
        Parameters
        ----------
        data : (str, str, float) tuple
            (name of stainer, message, time taken). Contains data to be used to create the History object
        row_map: {int: int} dictionary 
            Row mapping showing the relationship between the original and new
            row positions. Only applies to transformation for the specific
            stainer.
        col_map: {int: int} dictionary
            Column mapping showing the relationship between the original and
            new column positions. Only applies to transformation for the
            specific stainer.
        """
        self.history.append(History(data, row_map, col_map))
    
    def add_stainers(self, stain, use_orig_row = True, use_orig_col = True):
        """ Adds a stainer / list of stainers to current list of stainers to be executed.
        
        Parameters
        ----------
        stain : Stainer or Stainer list 
            stainers to be added to the DDF to be executed in the future
        use_orig_row : boolean, optional
            Indicates if indices in stainer refers to the initial dataframe, or
            the index of the dataframe at time of execution.
            If True, indices from initial dataframe are used. Defaults to True
        use_orig_col : boolean, optional
            Indicates if indices in stainer refers to the initial dataframe, or
            the index of the dataframe at time of execution.
            If True, indices from initial dataframe are used. Defaults to True
            
        Returns
        ----------
        ddf : DirtyDF
            Returns new copy of DDF with the stainer added
        """
        ddf = self.copy()
        if isinstance(stain, Stainer):
            ddf.stainers.append((stain, use_orig_row, use_orig_col))
        else:
            for st in stain:
                ddf.stainers.append((st, use_orig_row, use_orig_col))
        
        return ddf
        
    def reindex_stainers(self, new_order):
        """
        Reorder stainers in a specified order 
        
        Parameters
        ----------
        new_order : int list
            Indices of the new order of stainers. If original was [A, B, C] and
            new_order = [1, 2, 0], the resulting order will be [C, A, B].
        
        Returns
        ----------
        ddf : DirtyDF
            Returns new copy of DDF with the stainers rearranged
        """
        ddf = self.copy()
        ddf.stainers = list(map(lambda x: ddf.stainers[x], new_order))
        
        return ddf
    
    def shuffle_stainers(self):
        """
        Randomly reorder the stainers
        
        Returns
        ----------
        ddf : DirtyDF
            Returns new copy of DDF with the stainers rearranged
        """
        n = len(self.stainers)
        new_order = self.rng.choice([i for i in range(n)], size = n, replace = False)
        return self.reindex_stainers(new_order)
    
    def run_stainer(self, idx = 0):
        """
        Applies the transformation of the specified stainer
        
        Parameters
        ----------
        idx : int, optional
            Index of stainer to execute. Defaults to 0 (first stainer added)
            
        Returns
        ----------
        ddf : DirtyDF
            Returns new DDF after the specified stainer has been executed
        """
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
    
    def run_all_stainers(self):       
        """
        Applies the transformation of all stainers in order
            
        Returns
        ----------
        ddf : DirtyDF
            Returns new DDF after all the stainers have been executed
        """
        current_ddf = self
        for stainer in self.stainers:
            current_ddf = current_ddf.run_stainer()
        return current_ddf

    def copy(self):
        """
        Creates a copy of the DDF
        
        Returns
        ----------
        ddf : DirtyDF
            Returns copy of DDF
        """
        new_ddf = DirtyDF(self.df.copy(), copy = True)
        new_ddf.seed = self.seed
        new_ddf.rng = self.rng
        new_ddf.orig_shape = self.orig_shape
        new_ddf.stainers = self.stainers.copy()
        new_ddf.history = self.history.copy()
        new_ddf.row_map = self.row_map.copy()
        new_ddf.col_map = self.col_map.copy()
        return new_ddf
