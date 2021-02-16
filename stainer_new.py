from history import *

class Stainer:
    col_types = "all"
    
    def __init__(self, name, row_idx = [], col_idx = []):
        self.name = name
        self.row_idx = row_idx
        self.col_idx = col_idx
        
        self.__initialize_history__()
    
    def transform(self, df, rng):
        raise Exception("Stainer not implemented")
    
    def get_indices(self):
        return self.row_idx, self.col_idx
    
    def __initialize_history__(self):
        self.message = ""
    
    def update_history(self, message = None):
        self.message += message
    
    def get_history(self):
        """ Creates a history object and returns it"""
        return History(self.message)

# doesn't actually make use of the row/col so not very good example ._. 
class ShuffleStainer(Stainer):
    """ This isn't complete """ 
    col_types = "all"
    
    def __init__(self, name):
        super().__init__(name, [], [])
        
    def transform(self, df, rng, row = None, col = None):
        new_df = df.copy()
        
        if not row:
            row = self.row_idx
        if not col:
            col = self.col_idx
        
        new_df = new_df.sample(frac = 1)
        self.update_history("Shuffling")
        
        return new_df, {}, {}
