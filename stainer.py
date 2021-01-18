import pandas as pd
import numpy as np
from itertools import product
from random import choice, randint, random, sample
from random import seed as rseed
from time import time
from warnings import warn

from artificialdata import *
from errors import *

class Stainer():
    """
    Interface for the different types of stainers. 
    The transform method will take in dataframe and based on the information of the dataframe 
    and the parameters provided, it will select the appropriate columns for Staining
    
    Args:
        style (str):
            Type of Stainer
        deg (0-1):
            Level of staining to be applied. 0 = None; 1 = Maximum
        fixed_row (list):  
            List of row numbers (0-index) which are to be stained. 
            Defaults to None (no restriction on specific rows)
        fixed_col (list):  
            List of column names / numbers (0-index) which are to be stained. 
            Defaults to None (no additional restriction on specific columns)
        seed (int):
            Customise the start number of RNG. Overwrites seed stated in Combiner.
            Defaults to the Combiner seed if unstated
    """
    def __init__(self, style, deg, fixed_row = [], fixed_col = [], seed = None):
        self.style = style 
        if 0 <= deg <= 1:
            self.deg = deg
        else:
            raise InputError(f"Invalid value for degree: Should be in range [0, 1], provided {deg} instead", \
                             style, "Skipping Stainer")
        self.fixed_row = fixed_row
        self.fixed_col = fixed_col
        self.seed = seed
        
    def transform(self, ddf, seed):
        """ 
        Modifies the dirtydf object by applying the relevant staining 
        
        Args:
            ddf (dirtydf):
                DataFrame that is to be transformed using this stainer
            seed (int):
                Seed for the transformation
        """
        # Set seed
        if self.seed:
            seed = self.seed
        rseed(seed)
##        print("Seed set at: ", seed) # Line for debugging 

        
class AddDuplicate(Stainer):
    """
    Stainer that adds duplicate rows to dataset.
    
    Degree:
        Determines the approximate proportion of REMAINING (excluding the fixed_rows) data that would be duplicated
    
    Additional Args: 
        randomize_order (boolean):
            If True, data will be randomized after duplication to make it more difficult to spot 
            (Should be left False for time-series data)
        max_rep (2/3/4/5):
            Maximum number of times a row can appear after duplication. That is, if max_rep = 2, 
            the original row was duplicated once to create 2 copies total.
            Capped at 5 to conserve computational power
    """
    def __init__(self, deg, fixed_row = [], seed = None, randomize_order = False, max_rep = 2):
        super().__init__("Insert Duplicates", deg, fixed_row, [] , seed)
        self.randomize_order = randomize_order
        self.max_rep = max_rep
        if max_rep > 5:
            warn("Insert Duplicates: Max 5 duplicates per entry. Setting value to 5")
            self.max_rep = 5
        if max_rep <= 1:
            raise InputError("Invalid value for max_rep: Should be in range [2, 5], provided {max_rep} instead", \
                            style, "Skipping Stainer")

    def transform(self, ddf, seed):
        """
        A. For data that have been identified as fixed_rows, they will be duplicated regardless.
        B. For other data, there will be a % chance they will be duplicated, depending on the degree that was set
        
        For a data that is duplicated, it will be duplicated randomly between 2 and the max_rep specified.
        
        C. If randomizing is required after the adding of rows, the rows will be randomized.
        """
        super().transform(ddf, seed)
        
        temp_df = []
        start = time()
        initial_size = ddf.df.shape[0]
        counter = 0
        total_added = 0
        
        for row in ddf.df.itertuples(index = False):
            if counter in self.fixed_row: # STEP A
                temp_df.extend([list(row)] * randint(2, self.max_rep))
            else:
                rand_num = random() # STEP B
                if rand_num <= self.deg:
                    temp_df.extend([list(row)] * randint(2, self.max_rep))
                    total_added += 1
                else:
                    temp_df.append(list(row))
            counter += 1
        
        temp_df = pd.DataFrame(temp_df, columns = ddf.df.columns)
        
        if self.randomize_order: # STEP C 
            temp_df = temp_df.sample(frac=1).reset_index(drop=True)
                    
        ddf.df = temp_df

        final_size = ddf.df.shape[0]
        end = time()
        time_taken = end - start
        
        message = f"Added Duplicate Rows for {len(self.fixed_row)} specified rows and {total_added} additional rows. \n" + \
                  f"Each duplicated row should appear a maximum of {self.max_rep} times. \n" + \
                  f"Rows added: {final_size - initial_size}. \n"
        if ddf.history:
            hist = HistoryDF(message, time_taken, seed, ddf.df.copy())
        else:
            hist = HistoryDF(message, time_taken, seed)
        ddf.summary.append(hist)
        

class Nullify(Stainer):
    """
    Stainer that convert various values to missing data / values that represent missing values.

    Degree:
        Determines the proportion of selected data that would be nullified

    Additional Args:
        new_val (int/str):
            Value that would replace the specific data.
            Defaults to None
    """
    def __init__(self, deg, fixed_row = [], fixed_col = [], seed = None, new_val = None):
        super().__init__("Nullify", deg, fixed_row, fixed_col, seed)
        self.new_val = new_val

    def transform(self, ddf, seed):
        """
        If no specification, all rows and columns will be selected.
        The specified proportion of cells would then be identified to be nullified.
        """
        super().transform(ddf, seed)

        start = time()

        df = ddf.df
        n_rows, n_cols = df.shape

        rows, cols = self.fixed_row, self.fixed_col
        if not rows:
            rows = [i for i in range(n_rows)]
        if not cols:
            cols = [i for i in range(n_cols)]
            
        all_cells = list(product(rows, cols))
        total_null = int(len(all_cells) * self.deg)
        selected_cells = sample(all_cells, k = total_null)
        
        for row, col in selected_cells:
            df.iloc[row, col] = self.new_val
        
        end = time()
        time_taken = end - start
        
        message = f"Replaced {total_null} values to become {'empty' if self.new_val == None else self.new_val} in specificed rows/cols. \n"
        if ddf.history:
            hist = HistoryDF(message, time_taken, seed, ddf.df.copy())
        else:
            hist = HistoryDF(message, time_taken, seed)
        ddf.summary.append(hist)
        

"""
NOTE: Should there be an option to scale back to a certain range of values?
"""


class FunctionTransform(Stainer):
    """
    Stainer that will transform a numerical column into one with a different distribution
    
    Degree:
        Proportion of fixed_cols that will be transformed

    Additional Args:
        transformation (dictionary<function>):
            List of possible transformations that will be randomly selected to be applied to the columns.
            By default, the list of possible transformations will be set as
            {square: x**2, cube: x**3, inverse: 1/x, ln: ln(x)}
    """
    def __init__(self, deg, fixed_col = "all", seed = None, transformation = "random"):
        super().__init__("Variable Transform", deg, [], fixed_col, seed)
        self.trans = transformation
        if self.trans == "random":
            self.trans = {"square": lambda x: x**2,
                          "cube": lambda x: x**3,
                          "1/x": lambda x: 1000 if x == 0 else round(1/x, 2),
                          "ln": lambda x: 0 if x == 0 else round(np.log(x), 2)}

    def transform(self, ddf, seed):
        super().transform(ddf, seed)

        start = time()

        df = ddf.df
        n_cols = df.shape[1]

        if self.fixed_col == "all":
            cols = [i for i in range(n_cols)]

        rando_lst = sample(cols, int(n_cols * self.deg))

        message = ""
        
        for col in rando_lst:
            rando_func = choice(list(self.trans.keys()))
            df.iloc[:, col] = df.iloc[:, col].apply(self.trans[rando_func])
            message += f"Converted column {col} with transformation {rando_func}. \n"
    
        end = time()
        time_taken = end - start
        
        if ddf.history:
            hist = HistoryDF(message, time_taken, seed, ddf.df.copy())
        else:
            hist = HistoryDF(message, time_taken, seed)
        ddf.summary.append(hist)
        

        
        
        
        
        
        
