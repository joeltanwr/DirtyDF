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
# ShuffleStainer Example
# ^^^^^^^^^^^^^^^^^^^^^^^^^

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
# dataframe, as well as others. You may check with the ID column, or with the original dataframe above to verify that this is true.

# %%
# Furthermore, you may use the stainer's `get_history()` method to get the name of the stainer, a description of how the stainer 
# had transformed the dataframe, and the time taken for said transformation.

shuffle_stainer.get_history()

# %%
# InflectionStainer Example
# ^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# For this next example, we will be using a randomly generated dataset of 100 rows and 3 columns, an integer ID, and 2 animal class
# columns (this dataset has no 'meaning', it is simply for demo). In particular, we will demonstrate using the InflectionStainer to
# generate string inflections of the animal categories.

rng = np.random.default_rng(42) # reinitialize random generator
df2 = pd.DataFrame(zip(range(100), rng.choice(['Cat','Dog','Rabbit'], 100), rng.choice(['Cow', 'Sheep', 'Goat', 'Horse'], 100)),
                  columns=('id', 'class', 'class2'))

df2.head()

# %%
# Here are the distributions of the animal classes.
df2['class'].value_counts()
# %%
df2['class2'].value_counts()

# %%
# We inflect on the 2 animal columns (index 1 and 2), use only 3 inflection formats (original, lowercase, and pluralize), 
# and ignore inflections on the 'Dog' category in the first class and 'Cow' & 'Sheep' categories in the second class.
inflect_stainer = InflectionStainer(col_idx=[1, 2], num_format = 3, formats=['original', 'lowercase', 'pluralize'], 
                    ignore_cats={1: ['Dog'], 2: ['Cow', 'Sheep']})

new_df2, row_map2, col_map2 = inflect_stainer.transform(df2, rng)
new_df2.head()

# %%
# We can see the new distributions.
new_df2['class'].value_counts()
# %%
new_df2['class2'].value_counts()

# %%
# We can also check the description of the stainer's transform from its history (the 2nd element in the history tuple).
print(inflect_stainer.get_history()[1])

# %%
# For more info on each of the stainer's use-cases and input parameters, do check their respective documentations.