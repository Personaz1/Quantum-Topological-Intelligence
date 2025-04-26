import numpy as np

class PhaseCore:
    """
    Phase Core of the Difference Loop.
    Evaluates memory state stability based on variance.
    """
    def __init__(self, threshold=1.0):
        """
        :param threshold: float, variance threshold for determining stability
        """
        self.threshold = threshold

    def check_stability(self, memory_state):
        """
        Evaluates stability: if variance is above the threshold — returns False (unstable), otherwise True.
        :param memory_state: np.ndarray, current memory state
        :return: (bool, float) — is_stable, variance value
        """
        variance = np.var(memory_state)
        return variance < self.threshold, variance
