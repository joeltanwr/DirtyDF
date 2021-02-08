import pytest
import pandas as pd
import numpy as np
from mtjjv import DirtyDataFrame

#initialize test dataframes
df_num = pd.DataFrame([(0.1*i, 0) for i in range(6)])
df_cat = pd.DataFrame(
    [(0, 'Cat'), (1, 'Dog'), (2, 'Rabbit'), (3, 'Cat'), (4, 'Cat'), (5, 'Dog')],
    columns=('id', 'class')
    )

df_lst = [df_num, df_cat]

def test_ddf_instantiation():
    with pytest.raises(TypeError):
        DirtyDataFrame(0)
    assert DirtyDataFrame(df_num).org_df is df_num

@pytest.mark.parametrize('df', df_lst)
def test_ddf_inplaceness(df):
    ddf = DirtyDataFrame(df)
    ddf.attach_stainer(DuplicateRowStainer()) #check that .attach_stainer does not update in place
    assert ddf == DirtyDataFrame(df)

    ddf.attach_stainer(DuplicateRowStainer()).dirty(seed=0) #check that .dirty does not update in place
    assert ddf == DirtyDataFrame(df)

@pytest.mark.parametrize('df', df_lst)
def test_ddf_stainer_reproducibility(df):
    ddf = DirtyDataFrame(df).attach_stainer(DuplicateRowStainer())
    assert ddf.dirty(seed=0) == ddf.dirty(seed=0)

@pytest.mark.parametrize('df', df_lst)
def test_ddf_stainer_associativity(df):
    #check that gf(x) == g(f(x))
    ddf = DirtyDataFrame(df)
    assert ddf.attach_stainer(CategoricalInflectionStainer()).attach_stainer(DuplicateRowStainer()).dirty(seed=0) == \
        ddf.attach_stainer(CategoricalInflectionStainer()).dirty(seed=0).attach_stainer(DuplicateRowStainer()).dirty(seed=0)