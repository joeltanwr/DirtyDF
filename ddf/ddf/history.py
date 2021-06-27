class History:
    """ Stores the historical information of a stainer so that it can printed in an organised manner. """
    def __init__(self, data, row_map, col_map):
        """
        Constructor for History object
        
        Parameters
        ----------
        data : (str, str, float) tuple
            Stores important information about Stainer executed. Tuple contains (name of stainer, message, time taken)
        row_map: {int: int} dictionary 
            Row mapping showing the relationship between the original and new row positions. Only applies to transformation for the specific stainer.
        col_map: {int: int} dictionary
            Column mapping showing the relationship between the original and new column positions. Only applies to transformation for the specific stainer.
        """
        self.name, self.message, self.time = data
        self.row_map = row_map
        self.col_map = col_map

    def get_row_map(self):
        """
        Returns row mapping for the corresponding stainer
        
        Returns 
        ----------
        row_map : {int: int} dictionary 
            Mapping of previous row indices to row indices after stainer transformation.
        """
        return self.row_map

    def get_col_map(self):
        """
        Returns column mapping for the corresponding stainer
        
        Returns 
        ----------
        col_map : {int: int} dictionary 
            Mapping of previous column indices to column indices after stainer transformation.
        """
        return self.col_map

    def __str__(self):
        return f"{self.name} \n {self.message} \n Time taken: {self.time} \n"
