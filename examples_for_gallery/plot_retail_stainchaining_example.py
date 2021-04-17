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
# We now initiate our stainers.
