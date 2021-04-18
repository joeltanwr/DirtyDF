# -*- coding: utf-8 -*-
"""
Retail Dataset Example
======================

This page shows some typical use-cases of 'chainstaining' multiple stainers together to produce several distinct transformed
DirtyDFs, based on a retail dataset. We expect these types of procedures to be the most common use-case of this library.
"""
import pandas as pd
import numpy as np
from ddf.stainer import ShuffleStainer, InflectionStainer, NullifyStainer, DatetimeFormatStainer, DatetimeSplitStainer
from ddf.DirtyDF import DirtyDF

# %% 
# We load the dataset and view some basic dataset properties.
retail = pd.read_csv("../data/online_retail_small.csv", parse_dates = ["InvoiceDate"])
retail.info()

# %%
retail.head()

# %%
# Convert 'Country' column to 'category' type.
retail["Country"] = retail.Country.astype("category")

# %%
# We can stain the dataset in various ways; in particular, since there is a datetime component in this dataset,
# we can use the DatetimeFormatStainer and DatetimeSplitStainer.
# We can also add simple ShuffleStainer, NullifyStainer, and apply InflectionStainer on the countries as well.

# %%
# We first view the distribution of the Country column to see if inflection staining is applicable here.
retail.Country.value_counts()

# %%
# We can see that lowercase and uppercase inflections are applicable here, aside from the 'EIRE' category, which we can ignore.

# %%
# We now check the numeric distribution of the datetime column to see if datetime staining is applicable here.
retail.InvoiceDate.describe(datetime_is_numeric=True)

# %%
# We can see that the entire dataset consists of invoices within a month, and times are included.

# %%
# We now initiate our stainers. It is possible to change the name of the Stainer to reflect the output seen when
# printing the history
retail_ddf = DirtyDF(retail, seed = 42) # Create DDF
dt_split_stainer = DatetimeSplitStainer(name = "Date Split", keep_time = False) # Only split the date

# %%
# Since the col_type of the DatetimeSplitStainer is set to "datetime", it will automatically identify datetime columns
# and only execute the stainer on those columns. Note that this only applies when using a DDF. If using the stainer directly,
# the column number needs to be specified
retail_transformed = retail_ddf.add_stainers(dt_split_stainer).run_stainer()
retail_transformed.get_df().head()

new_retail_df, row_map, col_map = dt_split_stainer.transform(retail, np.random.default_rng(42), col_idx = [4])

# %%
# Since the DatetimeSpitStainer adds columns, we can check the column mapping to see how the columns were changed
retail_transformed.get_mapping(axis = 1) # or col_map if using the Stainer directly



