import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


class DirtyDataFrame:
    """A class to dirty data frames.

    df: A pandas DataFrame
    stainer_list: A list of stainers. Stainers are functions 
                  that take in a data frame and return a data 
                  frame. They could also take in a random number 
                  generator object.
    """

    def __init__(self, df):
        if isinstance(df, pd.DataFrame):
            self.org_df = df
        else:
            raise TypeError('df should be pandas DataFrame')

        self.stainer_list = []
 
    def dirty(self, stain_order=None, seed=42):
        pass

    def list_stainers(self):
        pass

    def attach_stainer(self, f0, **stainer_args):
        f1 = partial(f0, **stainer_args)
        self.stainer_list.append(f1)

    def check_order(self):
        pass

    def check_stainer(self, f):
        pass

    def describe_stainer(self, i):
        pass

    def summarise_stainers(self):
        pass


def add_duplicates(df, row_to_dup=None, times_to_dup= None, dup_at=None,
                   new_index='keep'):
    """Stainer to duplicate rows of a dataset.

    row_to_dup:  integer-based indexes to df
    times_to_dup: array like or integer. If array, it should have the same 
                  length as rows_to_dup.
    dup_at:       Keep at original position, or shuffle.
    new_index:    The new index to use for the new data frame, or to keep
                  duplicate values, or to use running integers. This can be 
                  'keep' or 'reindex'.
    """

    if row_to_dup is None:
        row_to_add = df.sample(n=1)
        row_to_dup = row_to_add.index[0]
        #new_df = pd.concat([df.iloc[:index_to_add, :], row_to_add, 
        #                    df.iloc[index_to_add:, :]])
    else:
        row_to_add = df.iloc[[row_to_dup]]
        
    new_df = pd.concat([df.iloc[:row_to_dup, :], row_to_add, 
                        df.iloc[row_to_dup:, :]])

    if new_index is not None:
        if new_index == 'keep':
            return(new_df)
        elif new_index == 'reindex':
            new_df.index = np.arange(new_df.shape[0])
        #else:
        #    new_df.index = new_index

    return(new_df)

def df_plot(pandas_df):
    # get shape
    df_shape = pandas_df.shape

    cell_colours = np.empty(df_shape, dtype=np.object)
    cell_colours[:] = 'xkcd:lightblue'

    #ax = plt.gca()
    #plt.gca()
    ax2 = plt.table(cellColours=cell_colours, loc='center')

    #ax2 = ax.table(cellColours=cell_colors, loc='center', colWidths=[0.5, 0.5])

    for cell in ax2.get_celld().values():
        cell.set_height(1.0/df_shape[0])
    ax2.axes.set_axis_off()
    return(ax2)
