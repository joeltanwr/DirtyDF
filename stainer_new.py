"""
To-do:
1. Fix initialisation of history so that it would reset upon creation of DDF
    - DONE

2A. AddDuplicate Stainer
2B. Nullify Stainer
2C. Function Stainer

3. Currently updating history is only message. Will need to adjust more if
want it to contain other information
"""

class Stainer:
    col_type = "all"
    
    def __init__(self, name, row_idx = [], col_idx = []):
        self.name = name
        self.row_idx = row_idx
        self.col_idx = col_idx
        
        self.__initialize_history__()

    def get_col_type(self):
        return self.col_type

    def get_indices(self):
        return self.row_idx, self.col_idx
    
    def transform(self, df, rng):
        raise Exception("Stainer not implemented")

    def init_transform(self, df, row, col):
        """
        Helper method to assign df / row / cols before transforming
        """
        new_df = df.copy()
        
        if not row:
            row = self.row_idx
        if not col:
            col = self.col_idx

        return new_df, row, col
                           
    # History-related methods #
    def __initialize_history__(self):
        self.message = ""
    
    def update_history(self, message = None):
        self.message += message
    
    def get_history(self):
        """ Creates a history object and returns it"""
        final = self.message
        self.__initialize_history__()
        return final

class ShuffleStainer(Stainer):
    """ This description isn't complete """ 
    col_type = "all"
    
    def __init__(self, name = "Shuffle"):
        super().__init__(name, [], [])
        
    def transform(self, df, rng, row = None, col = None):
        new_df, row, col = self.init_transform(df, row, col)

        # Shuffle + Create mapping
        new_df = new_df.sample(frac = 1, random_state = rng.bit_generator)
        original_idx = new_df.index
        new_df.reset_index(drop = True, inplace = True)
        new_idx = new_df.index
        row_map = dict(zip(original_idx, new_idx))
        
        self.update_history("Shuffling")
        
        return new_df, row_map, {}        

    




