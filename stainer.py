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

        
#wishful thinking: assumes some datetime equivalent for latlong has been coded.
class GeoFormatStainer(Stainer):
    """
    Stainer to alter the format of geospatial coordinates for given latlong columns.
    
    fixed_col:
        Columns to perform geo format staining on.
    num_format:
        Number of geo formats present within each column.
    formats:
        List of geo string format options that the GeoFormatStainer chooses from.
    """
    def __init__(self, fixed_col, seed = None, num_format = 2, formats = None):
        super().__init__("Staining geo formats", 0, [], fixed_col, seed)
        if formats:
            self.formats = formats
        else:
            self.formats = ["DMS", "MinDec", "DegDec"]
            
        if num_format > len(self.formats):
            raise InputError(f"Cannot have num_format be more than the number of formats options, which is {len(self.formats)}.")
        if num_format < 1:
            raise InputError(f"Cannot have num_format be less than 1.")
        self.num_format = num_format
        
    def transform(self, ddf, seed):
        super().transform(ddf, seed)
        
        temp_df = ddf.df.copy()
        start = time()
        nrow = temp_df.shape[0]
        
        #iterate over each column in fixed_col
        for col in self.fixed_col:
            # to-do: add error check for if col is of 'geo' type
            
            new_col = temp_df[col].copy() #initiate copy of col which will be used to replace the existing one
            random_idxs = np.array_split(np.random.choice(nrow, size=nrow, replace=False), num_format) #random indices to split dataframe by
            randomized_formats = random.shuffle(self.formats.copy()) #randomized list of formats
            
            for i in range(num_format):
                new_col.iloc[random_idxs[i]] = ddf.df[col].iloc[random_idxs[i],].apply(lambda x: x.strfgeo(randomized_formats[i]))
                #for each set of random indices, apply a different strfgeo format
            
            temp_df[col] = new_col
    
        ddf.df = temp_df

        end = time()
        time_taken = end - start
        
        message = f"Changed the format of the following geo columns: {fixed_col}. For each column, a maximum of {self.num_format} " + \
                  "formats were used."
        if ddf.history:
            hist = HistoryDF(message, time_taken, seed, ddf.df.copy())
        else:
            hist = HistoryDF(message, time_taken, seed)
        ddf.summary.append(hist)