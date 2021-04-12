# -*- coding: utf-8 -*-
"""
Basic Usage of Stainers (no DirtyDF)
====================================

This page shows some basic examples of using stainers to directly transform panda dataframes.
"""

import pandas as pd
import numpy as np
from ddf.stainer import ShuffleStainer, InflectionStainer, DatetimeFormatStainer, DatetimeSplitStainer
# %%
# For the first example, let us use a basic dataset containing only 6 rows and 2 columns, an integer ID and an animal class.

df = pd.DataFrame([(0, 'Cat'), (1, 'Dog'), (2, 'Rabbit'), (3, 'Cat'), (4, 'Cat'), (5, 'Dog')], columns=('id', 'class'))
df

# %%
# We now apply a ShuffleStainer to shuffle the rows in this dataset. Note that we require to pass in a numpy random generator for
# random generation.

# %%
# The stainer's transform method will output 3 objects: the transformed dataframe, a row map which maps the rows in the old dataframe
# to the new one, and a column map which maps the columns in the old dataframe to the new one.

shuffle_stainer = ShuffleStainer()
rng = np.random.default_rng(42)

new_df, row_map, col_map = shuffle_stainer.transform(df, rng)
new_df

# %%
# Also, we can check the row map to determine which rows in the old dataframe were mapped to the new ones. (Note that ShuffleStainer
# does not affect or alter columns, so the column map is simply an empty dictionary)

row_map

# %%
# The output shows that the 3rd row index (0-based indexing) from the original dataframe is mapped to the 0-th row in the new
# dataframe, as well as others. You may check with the original dataframe above to verify that this is true.

# %%
# Furthermore, you may use the stainer's `get_history()` method to get the name of the stainer, a description of how the stainer 
# had transformed the dataframe, and the time taken for said transformation.

shuffle_stainer.get_history()