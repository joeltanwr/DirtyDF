# -*- coding: utf-8 -*-
"""
Basic Usage of DirtyDF with Stainers
====================================

This page shows some basic examples of using DirtyDF, and applying stainers to transform them. We recommend you go through the 
Basic Usage of Stainers (no DirtyDF) example first.
"""
import pandas as pd
import numpy as np
from ddf.stainer import ShuffleStainer, InflectionStainer, RowDuplicateStainer
from ddf.DirtyDF import DirtyDF

# %%
# Single Stainer Example
# ^^^^^^^^^^^^^^^^^^^^^^

# %%
# For the first example, let us once again use the basic dataset containing only 6 rows and 2 columns, 
# an integer ID and an animal class.
animal = pd.DataFrame([(0, 'Cat'), (1, 'Dog'), (2, 'Rabbit'), (3, 'Cat'), (4, 'Cat'), (5, 'Dog')],
                  columns=('id', 'class'))

# %%
# Let us convert the pandas dataframe into a DirtyDF object. We specify a seed for the numpy random generator. This generator will
# be used for the staining.
animal_ddf = DirtyDF(animal, seed = 123)

# %%
# Let us use only 1 stainer: ShuffleStainer, for now.
shuffle_stainer = ShuffleStainer()

# %%
# Instead of calling on the stainer's transform method directly, we now add the stainer into the DirtyDF object, to be used later when
# calling the DDF.run_stainer() method.
animal_ddf2 = animal_ddf.add_stainers(shuffle_stainer)

# %%
# Note that the DDF methods return new DDF objects, and do not change the DDF in-place. This can be verified by checking the current
# stainers stored in a DDF using the .summarise_stainers() method.
animal_ddf.summarise_stainers() #empty

# %%
animal_ddf2.summarise_stainers() #ShuffleStainer present

# %%
# We run the stainer by calling the .run_stainer() method.
animal_ddf3 = animal_ddf2.run_stainer()

# %%
# Note that same as before, the above call returns a new DDF object. To view the dataframe content of the DDF object, we can use the
# .get_df() method.
animal_ddf3.get_df()

# %%
# Notice that animal_ddf2 still contains the original df, and contains ShuffleStainer inside, but not yet run.
animal_ddf2.get_df()

# %%
# On the other hand, since ShuffleStainer had already been run to obtain animal_ddf3, we can verify that animal_ddf3 does not contain
# ShuffleStainer anymore.
animal_ddf3.summarise_stainers() #empty

# %%
# We can view the history of stainers that were run to obtain animal_ddf3 (in this case, only the ShuffleStainer's history) by using
# the DDF.print_history() method.
animal_ddf3.print_history()

# %%
# We can also obtain the row and column mappings from the original df to the latest transformed df.
animal_ddf3.get_map_from_history(index=0, axis=0) #index=0 since there was only 1 stainer used, and axis=0 specifies rows.

# %%
animal_ddf3.get_map_from_history(index=0, axis=1) #axis=1 specifies columns. Note that ShuffleStainer doesn't alter columns.

# %%
# Multiple Stainers Example
# ^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Now lets get to the beauty of DirtyDF: using multiple stainers for transformation. For this example, we use 3 stainers, namely,
# ShuffleStainer, InflectionStainer, and RowDuplicateStainer.

shuffle_stainer = ShuffleStainer()
dup_stainer = RowDuplicateStainer(deg = 0.6, max_rep = 3)
inflection_stainer = InflectionStainer(num_format=2, formats=['lowercase', 'uppercase'])

# %%
# We work with the same dataset as before. However, note that we have to explicitly convert the 'class' column as 'category'
# type. This is for the InflectionStainer to be able to detect the column as a categorical and automatically be applied onto it.
animal["class"] = animal["class"].astype("category")

# %% 
# We can add multiple stainers at a time by passing a list of stainers into the .add_stainers()
# method.
animal_ddf_mult = DirtyDF(animal).add_stainers([shuffle_stainer, dup_stainer, inflection_stainer])

animal_ddf_mult.summarise_stainers()

# %%
# We can now run the stainers one-by-one by sequentially applying the .run_stainer() method. 

# %%
# .. note::
#   Stainers are run in the order that they were inserted in. This order can be altered by using the DDF.reindex_stainer() method,
#   or we can also shuffle the order of stainers by using the DDF.shuffle_stainer() method, however do note that not all stainers
#   are able to be run in any order (i.e. some stainers may need to come before or after others).

animal_ddf_mult2 = animal_ddf_mult.run_stainer().run_stainer().run_stainer()

# %%
# Note that we can also use .run_all_stainers() to run all stainers sequentially at once.
animal_ddf_mult3 = animal_ddf_mult.run_all_stainers() #does the same as above

# %%
animal_ddf_mult3.print_history()

# %%
# We can now view the transformed dataframe.
animal_ddf_mult3.get_df()

