import numpy as np

class Actor:
    """
    Актор Difference Loop.
    Перестраивает память в зависимости от режима (reset, amplify и др.).
    """
    def __init__(self, mode='reset'):
        """
        :param mode: str, режим действия (например, 'reset', 'amplify')
        """
        self.mode = mode

    def act(self, memory):
        """
        Перестраивает память: простейший вариант — сбросить или усилить деформацию.
        :param memory: Memory, объект памяти для модификации
        """
        if self.mode == 'reset':
            memory.state *= 0.5  # частичный сброс
        elif self.mode == 'amplify':
            memory.state *= 1.2  # усиление
        # Можно добавить другие режимы
