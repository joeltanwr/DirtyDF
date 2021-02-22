from time import time
import numpy as np
import pandas as pd

"""
To-do:
1. Fix initialisation of history so that it would reset upon creation of DDF
    - DONE

2A. AddDuplicate Stainer
2B. Nullify Stainer
2C. Function Stainer

3. Currently updating history is only message. Will need to adjust more if
want it to contain other information
    - Included time
"""

class Stainer:
    col_type = "all"
    
    def __init__(self, name = "Unnamed Stainer", row_idx = [], col_idx = []):
        self.name = name
        self.row_idx = row_idx
        self.col_idx = col_idx
        
        self.__initialize_history__()

    def get_col_type(self):
        return self.col_type

    def get_indices(self):
        return self.row_idx, self.col_idx
    
    def transform(self, df, rng, row_idx, col_idx):
        raise Exception("Stainer not implemented")

    def _init_transform(self, df, row_idx, col_idx):
        """
        Helper method to assign df / row / cols before transforming
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df should be pandas DataFrame')
        
        new_df = df.copy()
        
        if not row_idx:
            row_idx = self.row_idx
        if not col_idx:
            col_idx = self.col_idx

        return new_df, row_idx, col_idx
                           
    # History-related methods #
    def __initialize_history__(self):
        self.message = ""
        self.time = 0
    
    def update_history(self, message = "", time = 0):
        self.message += message
        self.time += time
    
    def get_history(self):
        """ Creates a history object and returns it"""
        msg, time = self.message, self.time
        if not time:
            time = "Time not updated. Use update_history to update time"
        self.__initialize_history__()
        return self.name, msg, time

class ShuffleStainer(Stainer):
    """ This description isn't complete """ 
    
    def __init__(self, name = "Shuffle"):
        super().__init__(name, [], [])
        
    def transform(self, df, rng, row_idx = None, col_idx = None):
        new_df, row_idx, col_idx = self._init_transform(df, row_idx, col_idx)

        start = time()
        # Shuffle + Create mapping
        new_df = new_df.sample(frac = 1, random_state = rng.bit_generator)
        original_idx = new_df.index
        new_df.reset_index(drop = True, inplace = True)
        new_idx = new_df.index

        row_map = np.zeros((df.shape[0], df.shape[0]))
        for i in range(len(row_map)):
            row_map[original_idx[i]][new_idx[i]] = 1
        
        end = time()
        
        self.update_history("Order of rows randomized", end - start)
        
        return new_df, row_map, {}        

class InflectionStainer(Stainer):
    """
    Stainer to introduce random inflections (capitalization, case format, pluralization) to given categorical columns.
    Parameters:
        name (str):
            Name of stainer.
        col_idx (int list):
            Columns to perform inflection stainer on. Must be specified.
        ignore_cats (str list / {int: str list} dict):
            Category strings to be ignored by stainer.
            If list: for all columns, ignore all categories present within the list.
            If dict: maps each col_idx to list of ignored category strings for that particular column.
        num_format (int):
            Number of inflection formats present within each column. If num_format > number of available formats, or num_format == -1, use all formats.
        formats (str list, or 'all'):
            List of inflection format options to chooses from. Choose from 'original', 'uppercase', 'lowercase', 'capitalize', 'camelize', 'pluralize', 
            'singularize', 'dasherize', 'humanize', 'titleize', and 'underscore'.
            If all inflection formats are desired, then input 'all'.
    """
    def __init__(self, col_idx, name="Inflection", ignore_cats = [], num_format = -1, formats = ['original', 'uppercase', 'lowercase']):
        super().__init__(name, [], col_idx)
        self.num_format = num_format
        
        if isinstance(ignore_cats, list): # convert from list to dict
            temp_ignore_cats = ignore_cats.copy()
            self.ignore_cats = {j: temp_ignore_cats for j in col_idx}
        elif isinstance(ignore_cats, dict):
            self.ignore_cats = ignore_cats
        else:
            raise TypeError("ignore_cats must be either a list of strings, or a dictionary mapping column index to list of strings")

        if formats == 'all':
            self.formats = ['original', 'uppercase', 'lowercase', 'capitalize', 'camelize', 'pluralize', 'singularize', 'dasherize', 'humanize',
                'titleize', 'underscore']
        else:
            self.formats = formats

    
    """
    Parameters:
        x (str):
            A category string.
        formats (str list):
            A list of string formats.
            
    Returns a set of strings, one for each inflection format in formats.
    """
    def get_inflected_strings(self, x, formats):
        import inflection

        default_format_map = {
            'original': lambda x: x,
            'uppercase': lambda x: x.upper(), 
            'lowercase': lambda x: x.lower(), 
            'capitalize': lambda x: x.capitalize(), 
            'camelize': inflection.camelize, 
            'pluralize': inflection.pluralize, 
            'singularize': inflection.singularize,
            'dasherize': inflection.dasherize, 
            'humanize': inflection.humanize, 
            'titleize': inflection.titleize, 
            'underscore': inflection.underscore}

        output_set = set()
        for format in formats:
            output_set.add(default_format_map[format](x))
        return output_set
        
    def transform(self, df, rng, row_idx = None, col_idx = None):
        new_df, row_idx, col_idx = self._init_transform(df, row_idx, col_idx)

        start = time()
        nrow = new_df.shape[0]
        
        #iterate over each column index
        for j in col_idx:
            new_col = df.iloc[:, j].copy() #instantiate a copy of this column which will be used to replace the existing one in new_df
            cats = [cat for cat in new_col.unique()] #unique categories in the column
            if self.num_format == -1 or self.num_format > len(self.formats):
                subformats = self.formats #use all formats
            else:
                subformats = rng.choice(self.formats, size=self.num_format, replace=False) #randomly select num_format formats from self.formats to be used for this column
            
            new_col_lst = []
            #interate over each string category
            for cat in cats:
                sub_col = new_col[new_col == cat] #subset of col which only contains this category
                

                if j in self.ignore_cats.keys() and cat in self.ignore_cats[j]: #ignore this category
                    new_col_lst.append(sub_col)
                else:
                    cat_inflection_formats = self.get_inflected_strings(cat, subformats) #set of inflected strings for this category
                    
                    new_col_lst.append(
                        pd.Series(rng.choice(list(cat_inflection_formats), size = sub_col.shape[0]), index=sub_col.index)
                    ) #randomly sample the strings from the available inflections    
            new_col = pd.concat(new_col_lst).reindex(new_col.index) #reindex based on original df index.
            new_df.iloc[:, j] = new_col
        
        end = time()
        self.update_history("Category inflections", end - start)
        return new_df, {}, {}

