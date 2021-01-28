import pandas as pd
import numpy as np


class Ddf(pd.DataFrame):
    _metadata = ['stainer_list']

    stainer_list = []

    @property
    def _constructor(self):
        return Ddf

 
    def my_func(self, str1):
        self.stainer_list.append(str1)
        print(str1)

    def dirty(self, stain_order=None, seed=42):
        pass

    def list_stainers(self):
        pass
