class History:
    def __init__(self, data, row_map, col_map):
        self.name, self.message, self.time = data
        self.row_map = row_map
        self.col_map = col_map

    def get_row_map(self):
        return self.row_map

    def get_col_map(self):
        return self.col_map

    def __str__(self):
        return f"{self.name} \n {self.message} \n Time taken: {self.time} \n"
