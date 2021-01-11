from random import shuffle
from errors import *

class DirtyDF():
    """
    Object that contains the dataframe, as well as other cruical information required 
    to perform staining more effectively.
    This contains information on the final DF that would be used.
    
    Args: 
        df (DataFrame):
            pandas DataFrame object that is meant to be dirtied
        history (boolean):
            If True, will store all intermediate DataFrame (May require signficantly more space + time)
            
    Attributes:
        summary (list<HistoryDF>):
            Contains the sequences and descriptions of each action. If history was set to True, will also
            store the DataFrame
        
    """
    def __init__(self, df, history = False, verbose = 1):
        self.df = df
        self.history = history
        self.summary = []
        
    def get_finalDF(self):
        return self.df
    
    def get_full_history(self):
        """
        Ideally, this would export some file that will contain the summary of the whole process.
        But for now, returns the actions of each of the actions taken
        """
        return list(map(lambda x: x.action, self.summary))
        
        

        
# If want to store any other useful attributes after each step, could do it here also
# E.g. If want to store time taken
class HistoryDF():
    """
    Stores the actions and dataframes after each action
    """
    
    def __init__(self, action, time, df = None):
        self.action = action
        self.df = df
        self.time = time
        
    def print_action(self, verbose = 0):
        """ 
        Prints a formatted summary of the action and other important statistics
        
        Args:
            verbose (0/1):
                0 to print out only the formatted words
                1 to include time taken
        """
        pass
        
        
class Combiner():
    """
    Combines the relevant stainers which are to be used for the dataset and applies the relevant 
    transformations in a pre-determined order
    
    Args:
        ddf (dirtydf): 
            DirtyDF object to be stained
        stainers (list<Stainers>):
            List of Stainers to be used
        random_order (boolean):
            By default, will be in a logical random order.
            If False, the stainers will be applied in sequence provided. However, the order provided must compatible
    """
    
    def __init__(self, ddf, stainers, random_order = True):
        self.ddf = ddf 
        self.stainers = stainers
        self.random_order = random_order
        
        if self.random_order:
            self.random_stain()
        else:
            if not self.check_valid(stainers):
                raise Exception("Invalid sequence of stainers")
    
    def get_finalDDF(self):
        return self.ddf
    
    def random_stain(self):
        """
        Rearranges the stainers in a random order
        
        !!! NOTE FOR FUTURE: Should have some logic to control the randomisation, 
            will need to insert
        """
        shuffle(self.stainers)
    
    def check_valid(self, lst):
        """
        Verifies that the sequence of stainers is logical and will be able to be executed
        
        Returns:
            True if valid sequence, else False
            
        !!! NOTE FOR FUTURE: This needs to be coded properly
        """
        return True
    
    def transform_all(self):
        for stain in self.stainers:
            try:
                stain.transform(self.ddf)
            except StainerNotImplementedError:
                print(f"{stain.style} not implemented")
#             except Exception as e: # Removed now to assist debugging
#                 print(f"ERROR: {e}. Skipping {stain.style}")
                
                
        
    