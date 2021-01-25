from random import shuffle
from random import seed as rseed
from time import time
from errors import *

class DirtyDF():
    """
    Object that contains the dataframe, as well as other cruical information required 
    to perform staining more effectively.
    This contains information on the final DF that would be used.
    
    Args: 
        df (DataFrame):
            pandas DataFrame object that is meant to be dirtied.
            Creates a new copy so that the original data will not be affected
        history (boolean):
            If True, will store all intermediate DataFrame (May require signficantly more space + time)
            
    Attributes:
        summary (list<HistoryDF>):
            Contains the sequences and descriptions of each action. If history was set to True, will also
            store the DataFrame        
    """
    def __init__(self, df, history = False, verbose = 1):
        self.df = df.copy()
        self.history = history
        self.summary = []
        
    def get_finalDF(self):
        return self.df

    def get_full_history(self):
        """
        Ideally, this would export some file that will contain the summary of the whole process.
        But for now, returns the actions of each of the actions taken
        """
        print("\n".join(list(map(lambda x: x.action, self.summary))))
        
        
# If want to store any other useful attributes after each step, could do it here also
# E.g. If want to store time taken
class HistoryDF():
    """
    Stores the actions and dataframes after each action
    """
    
    def __init__(self, action, time, seed, df = None):
        self.action = action
        self.df = df
        self.time = time
        self.seed = seed
        
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
        stainers (list<Stainers>):
            List of Stainers to be used
        random_order (boolean):
            By default, the stainers will be applied in sequence provided. However, the order provided must compatible
            If set to True, a logical random order will be chosen instead
        seed (int):
            Only relevant if random_order set to True
            Determines seed for order of stainers
        shuffle (boolean):
            Boolean to determine if row order should be randomized after all staining
    """
    
    def __init__(self, stainers, random_order = False, seed = None, shuffle = False):
        if seed:
            self.seed = seed
        else:
            self.seed = round(time() * 10)
        self.stainers = stainers
        self.random_order = random_order
        
        if self.random_order:
            self.random_stain()
        else:
            if not self.check_valid(stainers):
                raise Exception("Invalid sequence of stainers")

        self.shuffle = shuffle

    def get_ordering_seed(self):
        return self.seed
    
    def random_stain(self):
        rseed(self.seed)
        """
        Rearranges the stainers in a random order
        
        !!! NOTE FOR FUTURE: Should have some logic to control the randomisation, 
            will need to insert
        """
        try:
            shuffle(self.stainers)
            print(f"Randomisation complete with seed {self.get_ordering_seed()}")
        except:
            print("Randomisation failed. Attempting to use given order")
            if not self.check_valid(stainers):
                raise Exception("Invalid sequence of stainers")
    
    def check_valid(self, lst):
        """
        Verifies that the sequence of stainers is logical and will be able to be executed
        
        Returns:
            True if valid sequence, else False
            
        !!! NOTE FOR FUTURE: This needs to be coded properly
        """
        return True
    
    def transform_all(self, ddf, seed = None):
        """
        Processes all the Stainers and applies the transformations

        Args:
            ddf (dirtydf): 
                DirtyDF object to be stained
            seed (int):
                Default seed for all the Stainers within the Combiner
                If unspecified, will randomly generate a seed to be used

        Raises:
            StainerNotImplementedError
                If the specified stainer does not have a transform method.
                Will result in skipping the stainer
            Error
                If the stainer is not compatible with the dataset. Will result in
                skipping the stainer
        """
        if not seed:
            seed = round(time() * 10)
        
        for stain in self.stainers:
            try:
                stain.transform(ddf, seed)
            except StainerNotImplementedError:
                print(f"{stain.style} not implemented")
#             except Exception as e: # Removed now to assist debugging
#                 print(f"ERROR: {e}. Skipping {stain.style}")
        if self.shuffle:
            ddf.df = ddf.df.sample(frac=1).reset_index(drop=True)
        return ddf
