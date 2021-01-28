import pandas as pd
import numpy as np


class DirtyDataFrame:
    """An object to dirty data frames.

    df: A pandas DataFrame
    stainer_list: A list of stainers. Stainers are functions 
                  that take in a data frame and return a data 
                  frame.
    """

    def __init__(self, df, stainer_list=[]):
        self.org_df = df
        self.stainer_list = stainer_list
 
    def my_func(self, str1):
        self.stainer_list.append(str1)
        print(str1)

    def dirty(self, stain_order=None, seed=42):
        pass

    def list_stainers(self):
        pass

    def attach_stainer(self, f):
        pass
