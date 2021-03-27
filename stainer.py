from time import time
import numpy as np
import pandas as pd
from itertools import product

# require inflection module for inflection stainer
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

    @staticmethod
    def convert_mapper_dct_to_array(dct):
        '''
        Helper function to convert a mapper in dict form to numpy array. Useful when the final number of columns is unknown before stainer transformation.

        Parameters
        ----------
        dct: dictionary of {index: list of indices}
            the mapper in dict form

        Returns
        -------
        np.array
            the mapper in array form (required for Stainer output) 
        '''
        input_size = len(dct.keys())
        output_size = max([j for v in dct.values() for j in v]) + 1 #output size is the maximal index in dct values + 1 (zero-indexing)
        col_map = np.zeros((input_size, output_size)) #initialize column map array
        for i,v in dct.items():
            for j in v:
                col_map[i,j] = 1 #update column map array
        return col_map

class ShuffleStainer(Stainer):
    """ This description isn't complete """ 
    
    def __init__(self, name = "Shuffle"):
        super().__init__(name, [], [])
        
    def transform(self, df, rng, row_idx = None, col_idx = None):
        new_df, row_idx, col_idx = self._init_transform(df, row_idx, col_idx)

        start = time()
        # Shuffle + Create mapping
        new_df["_extra_index_for_stainer"] = range(df.shape[0])
        new_df = new_df.sample(frac = 1, random_state = rng.bit_generator)
        new_idx = new_df["_extra_index_for_stainer"].tolist()
        new_df.drop("_extra_index_for_stainer", axis = 1, inplace = True)
        
        row_map = np.zeros((df.shape[0], df.shape[0]))
        for i in range(len(row_map)):
            row_map[new_idx[i]][i] = 1
        end = time()
        
        self.update_history("Order of rows randomized", end - start)
        
        return new_df, row_map, {}        

class RowDuplicateStainer(Stainer):
    """
    Stainer to duplicate rows of a dataset.
    deg (0, 1]:
        Proportion of given data that would be duplicated.
        Note: If 5 rows were specified and deg = 0.6, only 3 rows will be duplicated
    max_rep (2/3/4/5):
        Maximum number of times a row can appear after duplication. That is, if max_rep = 2, 
        the original row was duplicated once to create 2 copies total.
        Capped at 5 to conserve computational power
    """  
    def __init__(self, deg, max_rep = 2, name = "Add Duplicates", row_idx = []):
        super().__init__(name, row_idx, [])
        if deg <= 0 or deg > 1:
            raise ValueError("Degree should be in range (0, 1]")
        self.deg = deg
        if max_rep not in [2, 3, 4, 5]:
            raise ValueError("max_rep should be in range [2, 5]")
        self.max_rep = max_rep

    def transform(self, df, rng, row_idx = None, col_idx = None):
        _, row, col = self._init_transform(df, row_idx, col_idx)
        new_df = []
        row_map = np.zeros((df.shape[0], int(df.shape[0] * self.deg * self.max_rep) + 1))
        
        start = time()
        idx_to_dup = rng.choice(row, size = int(self.deg * df.shape[0]), replace = False)
        idx_to_dup.sort()
        
        new_idx = 0

        for old_idx, *row in df.itertuples():
            if len(idx_to_dup) and old_idx == idx_to_dup[0]:
                num_dup = rng.integers(2, self.max_rep, endpoint = True)
                new_df.extend([list(row)] * num_dup)
                for i in range(num_dup):
                    row_map[old_idx][new_idx] = 1
                    new_idx += 1
                idx_to_dup = idx_to_dup[1:]
            else:
                new_df.append(list(row))
                row_map[old_idx][new_idx] = 1
                new_idx += 1

        row_map = row_map[:, :new_idx]
        
        new_df = pd.DataFrame(new_df, columns = df.columns)
        end = time()
        
        message = f"Added Duplicate Rows for {int(self.deg * df.shape[0])} rows. \n" + \
                  f"  Each duplicated row should appear a maximum of {self.max_rep} times. \n" + \
                  f"  Rows added: {new_df.shape[0] - df.shape[0]}"
        
        self.update_history(message, end - start)
        
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
        formats (str list, or None):
            List of inflection format options to chooses from. Choose from 'original', 'uppercase', 'lowercase', 'capitalize', 'camelize', 'pluralize', 
            'singularize', 'dasherize', 'humanize', 'titleize', and 'underscore'.
            If None, all inflections are used.
    """
    def __init__(self, col_idx, name="Inflection", ignore_cats = [], num_format = -1, formats = None):
        super().__init__(name, [], col_idx)
        self.num_format = num_format
        
        if isinstance(ignore_cats, list): # convert from list to dict
            temp_ignore_cats = ignore_cats.copy()
            self.ignore_cats = {j: temp_ignore_cats for j in col_idx}
        elif isinstance(ignore_cats, dict):
            self.ignore_cats = ignore_cats
        else:
            raise TypeError("ignore_cats must be either a list of strings, or a dictionary mapping column index to list of strings")

        if formats:
            self.formats = formats
        else:
            self.formats = ['original', 'uppercase', 'lowercase', 'capitalize', 'camelize', 'pluralize', 'singularize', 'dasherize', 'humanize',
                'titleize', 'underscore'] #default formats; 10 total inflections + original

    
    """
    Parameters:
        x (str):
            A category string.
        formats (str list):
            A list of string formats.
            
    Returns a set of strings, one for each inflection format in formats.
    """
    def _get_inflected_strings(self, x, formats):
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
                    cat_inflection_formats = self._get_inflected_strings(cat, subformats) #set of inflected strings for this category
                    
                    new_col_lst.append(
                        pd.Series(rng.choice(list(cat_inflection_formats), size = sub_col.shape[0]), index=sub_col.index)
                    ) #randomly sample the strings from the available inflections    
            new_col = pd.concat(new_col_lst).reindex(new_col.index) #reindex based on original df index.
            new_df.iloc[:, j] = new_col
        
        end = time()
        self.update_history("Category inflections", end - start)
        return new_df, {}, {}

class DateFormatStainer(Stainer):
    """
    Stainer to alter the format of dates for given date columns.
    
    Parameters:
        name (str):
            Name of stainer.
        col_idx (int list):
            Columns to perform date stainer on. Must be specified.
        num_format (int):
            Number of date formats present within each column. If num_format > number of available formats, or num_format == -1, use all formats.
        formats (str list or None):
            List of date string format options that the DateFormatStainer chooses from. Use datetime module string formats (e.g. '%d%b%Y'). If None,
            a default list of 41 non-ambiguous (month is named) date formats are provided.
    """
    def __init__(self, col_idx, name="Date Formats", num_format = 2, formats = None):
        import itertools
        
        super().__init__(name, [], col_idx)
        self.num_format = num_format

        if formats:
            self.formats = formats
        else:
            self.formats = [f"{dm_y[0]}{br}{dm_y[1]}" for br in [",", ", ", "-", "/", " "]
                                for m_type in ["%b", "%B"]
                                for d_m in itertools.permutations(["%d", m_type])
                                for d_m_str in [f"{d_m[0]}{br}{d_m[1]}"]
                                for dm_y in itertools.permutations([d_m_str, '%Y'])
                           ] + ['%Y%m%d'] #default formats; 41 total and non-ambiguous
            
        
    def transform(self, df, rng, row_idx = None, col_idx = None):
        new_df, row_idx, col_idx = self._init_transform(df, row_idx, col_idx)

        start = time()
        nrow = new_df.shape[0]
        
        #iterate over each column index
        for j in col_idx:
            new_col = df.iloc[:, j].copy() #instantiate a copy of this column which will be used to replace the existing one in new_df
            if self.num_format == -1 or self.num_format > len(self.formats):
                subformats = self.formats #use all formats
            else:
                subformats = rng.choice(self.formats, size=self.num_format, replace=False) #randomly select num_format formats from self.formats to be used for this column
            
            random_idxs = np.array_split(rng.choice(nrow, size=nrow, replace=False), len(subformats)) #randomly split dataframe indices into len(subformats) number of groups
            
            for i in range(len(subformats)): #for each group of indices, apply a different format from subformats
                new_col.iloc[random_idxs[i]] = new_df.iloc[random_idxs[i], j].apply(lambda x: x.strftime(subformats[i]))
                #for each set of random indices, apply a different strftime format

            new_df.iloc[:, j] = new_col
    
        end = time()
        self.update_history("Date Formats", end - start)
        return new_df, {}, {}

class DateSplitStainer(Stainer):
    """
    Stainer that splits each given date / datetime columns into 3 columns respectively, representing day, month, and year. 
    If a given column's name is 'X', then the respective generated column names are 'X_day', 'X_month', and 'X_year'.
    If a column is split, the original column will be dropped.
    For 'X_month' and 'X_year', a format from ['m', '%B', '%b'], and ['%Y', '%y'] is randomly chosen respectively. 
    
    Parameters:
        name (str):
            Name of stainer.
        col_idx (int list):
            date columns to perform date splitting on. Must be specified.
        prob:
            probability that the stainer splits a date column. Probabilities of split for each given date column are independent.
    """
    def __init__(self, col_idx, name="Date Split", prob=1.0):
        super().__init__(name, [], col_idx)

        if prob < 0 or prob > 1:
            raise ValueError("prob is a probability, it must be in the range [0, 1].")
        else:
            self.prob = prob
        
    def transform(self, df, rng, row_idx = None, col_idx = None):
        new_df, row_idx, col_idx = self._init_transform(df, row_idx, col_idx)

        start = time()
        
        message = f"Split the following date columns: "
        
        col_map_dct = {j: [] for j in range(df.shape[1])} #initialize column map dictionary; new number of columns is unknown at start.
        j_new = 0 #running column index for output df

        #iterate over all columns, and apply logic only when current column index is in self.col_idx
        for j in range(df.shape[1]):
            if (j not in self.col_idx) or (rng.random() > self.prob): #current column index not in self.col_idx, or no split due to probability
                col_map_dct[j].append(j_new)
                j_new += 1
            else:
                col_name = df.columns[j]
                message += f"{col_name}, "
                
                #check to ensure no undetected column name conflict
                if f"{col_name}_day" in new_df.columns:
                    raise KeyError(f"column name: '{col_name}_day' already exists in dataframe.")
                if f"{col_name}_month" in new_df.columns:
                    raise KeyError(f"column name: '{col_name}_month' already exists in dataframe.")
                if f"{col_name}_year" in new_df.columns:
                    raise KeyError(f"column name: '{col_name}_year' already exists in dataframe.")
                
                month_format = rng.choice(["%m", "%B", "%b"]) #randomly chosen month format
                year_format = rng.choice(["%Y", "%y"]) #randomly chosen year format

                new_df.drop(col_name, axis=1, inplace=True)
                new_df.insert(j_new, f"{col_name}_day", df[col_name].apply(lambda x: x.strftime("%d")))
                new_df.insert(j_new + 1, f"{col_name}_month", df[col_name].apply(lambda x: x.strftime(month_format)))
                new_df.insert(j_new + 2, f"{col_name}_year", df[col_name].apply(lambda x: x.strftime(year_format)))
                
                col_map_dct[j].extend([j_new, j_new + 1, j_new + 2])
                j_new += 3
        
        if j == j_new - 1:
            message = "No date columns were split."
        else:
            message = message[:-2]

        col_map = Stainer.convert_mapper_dct_to_array(col_map_dct)

        end = time()
        self.update_history(message, end - start)
        return new_df, {}, col_map

class BinningStainer(Stainer):
    """
    Stainer that bins continuous columns into discrete groups (each group represents an interval [a,b)).
    
            Columns to perform binning on. Must be specified.
        group_size:
            Number of elements in each interval group.
        n_groups:
            Number of groups to bin to. Ignored if either range or size is not None.
        sf:
            Number of significant digits to be used in the output string representation for the intervals.
    """
    def __init__(self, col_idx, name="Binning", group_size=None, n_groups=5, sf=4):
        super().__init__(name, [], col_idx)
        self.type = type
        self.range = range
        self.group_size = group_size
        self.n_groups = n_groups
        self.sf = sf
    
    @staticmethod
    def _bin_into_group(x, cutpoints):
        """
        Helper to bin decimal into the correct group.
        """
        #binary search for upper bound index, which is 'high'
        low=0
        high=len(cutpoints)-1
        while low < high:
            mid = low + (high - low) // 2
            if x < cutpoints[mid]:
                high = mid
            else:
                low = mid + 1

        lower_bound = cutpoints[high - 1]
        upper_bound = cutpoints[high]

        if high == len(cutpoints) - 1: #last index, closed interval
            return f"[{lower_bound}, {upper_bound}]"
        else: #not last index, half-open interval
            return f"[{lower_bound}, {upper_bound})"


    def transform(self, df, rng, row_idx=None, col_idx=None):
        new_df, row_idx, col_idx = self._init_transform(df, row_idx, col_idx)

        start = time()

        #helper function to round to significant digits
        def round_sig(x, sig=2):
            return round(x, sig-int(np.floor(np.log10(abs(x))))-1)

        #iterate over each column index
        for j in col_idx:
            new_col = df.iloc[:, j].copy() #instantiate a copy of this column which will be used to replace the existing one in new_df

            if self.group_size:
                n_groups = new_col.shape[0] // self.group_size
            else:
                n_groups = self.n_groups
            
            cutpoints = [round_sig(cp, self.sf) for cp in new_col.quantile([i/n_groups for i in range(n_groups + 1)], interpolation='lower').tolist()]

            new_df.iloc[:, j] = new_col.apply(lambda x: x if pd.isna(x) else self._bin_into_group(x, cutpoints))
        
        end = time()
        self.update_history("Binning", end - start)
        return new_df, {}, {}

      
from latlong import Latlong

class LatlongFormatStainer(Stainer):
    """
    Stainer to alter the format of datetimes for given latlong columns.
    
    Parameters:
        name (str):
            Name of stainer.
        col_idx (int list):
            Columns to perform latlong stainer on. Must be specified.
        num_format (int):
            Number of latlong formats present within each column. If num_format > number of available formats, or num_format == -1, use all formats.
        formats (str list or None):
            List of latlong string format options that the LatlongFormatStainer chooses from. Use the Latlong module string formats. 
            If None, a default list of formats are provided.
    """
    def __init__(self, col_idx, name="Latlong Formats", num_format = 2, formats = None):
        import itertools
        
        super().__init__(name, [], col_idx)
        self.num_format = num_format

        if formats:
            self.formats = formats
        else:
            self.formats = ['DMS', 'MinDec'] #default formats
            
        
    def transform(self, df, rng, row_idx = None, col_idx = None):
        new_df, row_idx, col_idx = self._init_transform(df, row_idx, col_idx)

        start = time()
        nrow = new_df.shape[0]
        
        #iterate over each column index
        for j in col_idx:
            new_col = df.iloc[:, j].copy() #instantiate a copy of this column which will be used to replace the existing one in new_df
            if self.num_format == -1 or self.num_format > len(self.formats):
                subformats = self.formats #use all formats
            else:
                subformats = rng.choice(self.formats, size=self.num_format, replace=False) #randomly select num_format formats from self.formats to be used for this column
            
            random_idxs = np.array_split(rng.choice(nrow, size=nrow, replace=False), len(subformats)) #randomly split dataframe indices into len(subformats) number of groups
            
            for i in range(len(subformats)): #for each group of indices, apply a different format from subformats
                new_col.iloc[random_idxs[i]] = new_df.iloc[random_idxs[i], j].apply(lambda x: x if pd.isna(x) else x.strflatlong(subformats[i]))
                #for each set of random indices, apply a different latlong format

            new_df.iloc[:, j] = new_col
    
        end = time()
        self.update_history("Latlong Formats", end - start)
        return new_df, {}, {}


class LatlongSplitStainer(Stainer):
    """
    Stainer that splits each given latlong columns into 6 columns, representing degree, minute, and seconds, for lat and long respectively.
    If a given column's name is 'X', then the respective generated column names 'X_lat_deg', 'X_lat_min', 'X_lat_sec', 'X_long_deg', 'X_long_min',
    and 'X_long_sec'.
    If a column is split, the original column will be dropped.
    
    Parameters:
        name (str):
            Name of stainer.
        col_idx (int list):
            latlong columns to perform latlong splitting on. Must be specified.
        prob:
            probability that the stainer splits a latlong column. Probabilities of split for each given date column are independent.
    """
    def __init__(self, col_idx, name="Latlong Split", prob=1.0):
        super().__init__(name, [], col_idx)

        if prob < 0 or prob > 1:
            raise ValueError("prob is a probability, it must be in the range [0, 1].")
        else:
            self.prob = prob
        
    def transform(self, df, rng, row_idx = None, col_idx = None):
        new_df, row_idx, col_idx = self._init_transform(df, row_idx, col_idx)

        start = time()
        
        message = f"Split the following latlong columns: "
        
        col_map_dct = {j: [] for j in range(df.shape[1])} #initialize column map dictionary; new number of columns is unknown at start.
        j_new = 0 #running column index for output df

        #iterate over all columns, and apply logic only when current column index is in self.col_idx
        for j in range(df.shape[1]):
            if (j not in self.col_idx) or (rng.random() > self.prob): #current column index not in self.col_idx, or no split due to probability
                col_map_dct[j].append(j_new)
                j_new += 1
            else:
                col_name = df.columns[j]
                message += f"{col_name}, "
                
                #check to ensure no undetected column name conflict
                for suffix in ['lat_deg', 'lat_min', 'lat_sec', 'long_deg', 'long_min', 'long_sec']:
                    if f"{col_name}_{suffix}" in new_df.columns:
                        raise KeyError(f"column name: '{col_name}_{suffix}' already exists in dataframe.")

                new_df.drop(col_name, axis=1, inplace=True)
                new_df.insert(j_new, f"{col_name}_lat_deg", df[col_name].apply(lambda x: x if pd.isna(x) else x.strflatlong("%da")))
                new_df.insert(j_new + 1, f"{col_name}_lat_min", df[col_name].apply(lambda x: x if pd.isna(x) else x.strflatlong("%ma")))
                new_df.insert(j_new + 2, f"{col_name}_lat_sec", df[col_name].apply(lambda x: x if pd.isna(x) else x.strflatlong("%s5a")))
                new_df.insert(j_new + 3, f"{col_name}_long_deg", df[col_name].apply(lambda x: x if pd.isna(x) else x.strflatlong("%do")))
                new_df.insert(j_new + 4, f"{col_name}_long_min", df[col_name].apply(lambda x: x if pd.isna(x) else x.strflatlong("%mo")))
                new_df.insert(j_new + 5, f"{col_name}_long_sec", df[col_name].apply(lambda x: x if pd.isna(x) else x.strflatlong("%s5o")))
                
                col_map_dct[j].extend([j_new, j_new + 1, j_new + 2, j_new + 3, j_new + 4, j_new + 5])
                j_new += 6
        
        if j == j_new - 1:
            message = "No latlong columns were split."
        else:
            message = message[:-2]

        col_map = Stainer.convert_mapper_dct_to_array(col_map_dct)

        end = time()
        self.update_history(message, end - start)
        return new_df, {}, col_map