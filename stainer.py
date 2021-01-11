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

     
