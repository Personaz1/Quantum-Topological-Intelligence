import numpy as np

class PhaseCore:
    """
    Фазовое ядро Difference Loop.
    Оценивает устойчивость состояния памяти по дисперсии.
    """
    def __init__(self, threshold=1.0):
        """
        :param threshold: float, порог дисперсии для определения устойчивости
        """
        self.threshold = threshold

    def check_stability(self, memory_state):
        """
        Оценивает устойчивость: если дисперсия выше порога — возвращает False (нестабильно), иначе True.
        :param memory_state: np.ndarray, текущее состояние памяти
        :return: (bool, float) — устойчиво ли, значение дисперсии
        """
        variance = np.var(memory_state)
        return variance < self.threshold, variance
