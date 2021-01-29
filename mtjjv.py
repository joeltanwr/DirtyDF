import pandas as pd
import numpy as np


class DirtyDataFrame:
    """A class to dirty data frames.

    df: A pandas DataFrame
    stainer_list: A list of stainers. Stainers are functions 
                  that take in a data frame and return a data 
                  frame.
    """

    def __init__(self, df, stainer_list=[]):
        if isinstance(df, pd.DataFrame):
            self.org_df = df
        else:
            raise TypeError('df should be pandas DataFrame')

        self.stainer_list = stainer_list
 
    def dirty(self, stain_order=None, seed=42):
        pass

    def list_stainers(self):
        pass

    def attach_stainer(self, f):
        pass

    def check_order(self):
        pass

    def check_stainer(self, f):
        pass

    def describe_stainer(self, i):
        pass

    def summarise_stainers(self):
        pass


def add_duplicates(df, rows_to_dup=None, times_to_dup= None, dup_at=None,
                   new_index=None):
    """Stainer to duplicate rows of a dataset.

    rows_to_dup:  integer-based indexes to df
    times_to_dup: array like or integer. If array, it should have the same 
                  length as rows_to_dup.
    dup_at:       Keep at original position, or shuffle.
    new_index:    The new index to use for the new data frame, or to keep
                  duplicate values, or to use running integers.
    """

    if rows_to_dup is None:
        pass
        

