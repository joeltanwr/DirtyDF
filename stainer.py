import pandas as pd
from random import randint, random
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
    """
    def __init__(self, style, deg, fixed_row = [], fixed_col = []):
        self.style = style 
        if 0 <= deg <= 1:
            self.deg = deg
        else:
            raise InputError(f"Invalid value for degree: Should be in range [0, 1], provided {deg} instead", \
                             style, "Skipping Stainer")
        self.fixed_row = fixed_row
        self.fixed_col = fixed_col
        
    def transform(self, ddf):
        """ 
        Modifies the dirtydf object by applying the relevant staining 
        
        Args:
            ddf (dirtydf):
                DataFrame that is to be transformed using this stainer
        """
        raise StainerNotImplementedError("Transform not defined", "General")

        
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
    def __init__(self, deg, fixed_row = [], fixed_col = [], randomize_order = False, max_rep = 2):
        super().__init__("Insert Duplicates", deg, fixed_row, fixed_col)
        if fixed_col:
            warn("Insert Duplicates: Cannot fix column. Using all columns instead")
            fixed_col = []
        self.randomize_order = randomize_order
        self.max_rep = max_rep
        if max_rep > 5:
            warn("Insert Duplicates: Max 5 duplicates per entry. Setting value to 5")
            self.max_rep = 5
        if max_rep <= 1:
            raise InputError("Invalid value for max_rep: Should be in range [2, 5], provided {max_rep} instead", \
                            style, "Skipping Stainer")

    def transform(self, ddf):
        """
        A. For data that have been identified as fixed_rows, they will be duplicated regardless.
        B. For other data, there will be a % chance they will be duplicated, depending on the degree that was set
        
        For a data that is duplicated, it will be duplicated randomly between 2 and the max_rep specified.
        
        C. If randomizing is required after the adding of rows, the rows will be randomized.
        """
        temp_df = []
        start = time()          
        counter = 0
        
        for row in ddf.df.itertuples(index = False):
            if counter in self.fixed_row: # STEP A
                temp_df.extend([list(row)] * randint(2, self.max_rep))
            else:
                rand_num = random() # STEP B
                if rand_num <= self.deg:
                    temp_df.extend([list(row)] * randint(2, self.max_rep))
                else:
                    temp_df.append(list(row))
            counter += 1
        
        temp_df = pd.DataFrame(temp_df, columns = ddf.df.columns)
        
        if self.randomize_order: # STEP C 
            temp_df = temp_df.sample(frac=1).reset_index(drop=True)
                    
        ddf.df = temp_df
        
        end = time()
        time_taken = end - start
        
        message = f"Added Duplicate Rows for {len(self.fixed_row)} specified rows and {self.deg * 100}% of the remaining rows. Each duplicated row should appear a maximum of {self.max_rep} time"
        if ddf.history:
            hist = HistoryDF(message, time_taken, ddf.df.copy())
        else:
            hist = HistoryDF(message, time_taken)
        ddf.summary.append(hist)
        
        
        