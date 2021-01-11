class Error(Exception):
    pass

class InputError(Error):
    """
    Invalid input by user
    
    Attributes:
        message: Message to be shown to user
        name: Name of stainer resulting in the problem
        next_action: What the program will continue to do next (e.g. skip / terminate)
    """
    def __init__(self, message, name, next_action):
        self.message = message
        self.name = name
        self.next_action = next_action    
        
    def __str__(self):
        return f"{self.name}: {self.message}. Next action: {self.next_action}"

class StainerNotImplementedError(Error):
    """
    Error when the transform method has not been defined
    
    Attributes:
        message: message to be shown to user
        name: name of stainer not implemented
    """
    def __init__(self, message, name):
        self.message = message
        self.name = name