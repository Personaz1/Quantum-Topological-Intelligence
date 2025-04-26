import numpy as np

class Actor:
    """
    Actor of the Difference Loop.
    Restructures memory depending on the mode (reset, amplify, etc.).
    """
    def __init__(self, mode='reset'):
        """
        :param mode: str, action mode (e.g., 'reset', 'amplify')
        """
        self.mode = mode

    def act(self, memory):
        """
        Restructures memory: the simplest option is to reset or amplify deformation.
        :param memory: Memory, memory object to modify
        """
        if self.mode == 'reset':
            memory.state *= 0.5  # partial reset
        elif self.mode == 'amplify':
            memory.state *= 1.2  # amplification
        # Other modes can be added
