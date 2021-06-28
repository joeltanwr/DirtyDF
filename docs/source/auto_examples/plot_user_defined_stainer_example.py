# -*- coding: utf-8 -*-
"""
User-defined Custom Stainers
============================

This page shows an example of how to create your own user-defined custom stainers, which subclasses from the Stainer class.
"""
from ddf.stainer import Stainer

# %%
# When creating a new stainer, it needs to inherit from the Stainer class in
# ddf.stainer.

#################################################################################
# The initialisation should include the name, row indices (if applicable) and
# column indices (if applicable). If any other relevant initialisations are
# required, they can be included as well.
# If the row or column indices do not apply to the stainer,
# an empty list can be provided to the superclass init.

#################################################################################
# When defining the transform, the parameters to be included should be
# df (the dataframe to be transformed), rng (a RNG Generator), row_idx = None and
# col_idx = None.

#################################################################################
# In the transform method, the self._init_transform(df, row_idx, col_idx) method
# can be called to accurately generate the row_idx and col_idx (This allows
# the Stainer to work correctly with DDF). The transform method should then
# implement the Stainer.

#################################################################################
# To provide relevant statistics to the user, messages and timings can be added.
# These can be added via self.update_history(message, time)

#################################################################################
# A row mapping and column mapping are also required. These represent a movement
# or creation of any row / col in the dataframe. It should be formatted as a
# dictionary where key = old_row/col_index, value = List of indices of where the 
# corresponding row/col ended up in the new dataframe. For instance, if the 
# index-2 row was duplicated and is now the index-2 and index-3 row,
# the row_map should contain the entry {2: [2, 3]}. If the rows/columns order
# were not altered, an empty dictionary should be returned.

#################################################################################
# The transform function should return a tuple of the new dataframe, row mapping,
# and the column mapping.

#################################################################################
# Refer to the sample code below for an example. 

class ShuffleStainer(Stainer):
    def __init__(self, name = "Shuffle"):   
        super().__init__(name, [], [])  # name, row_idx, col_idx
        
    def transform(self, df, rng, row_idx = None, col_idx = None):
        new_df, row_idx, col_idx = self._init_transform(df, row_idx, col_idx)

        start = time()

        """
        ## Implementation of Shuffle ##
        Creates new_df
        Creates new_idx = Original row numbers, in order of the new row numbers
        """
        pass 


        """Creates the mapping"""
        row_map = {}
        for i in range(df.shape[0]):
            row_map[new_idx[i]] = [i]

        end = time() # Timer to be added into history
        self.update_history("Order of rows randomized", end - start) # message, time
        return new_df, row_map, {} # new dataframe, row map, column map
