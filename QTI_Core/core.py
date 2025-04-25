from .sensor import Sensor
from .memory import Memory
from .phase import PhaseCore
from .actor import Actor

class QTI_Core:
    """
    Основной цикл Difference Loop: S (Sensor) → M (Memory) → Φ (PhaseCore) → A (Actor) → S
    Инкапсулирует всю архитектуру QTI и реализует один шаг петли различий.
    """
    def __init__(self, input_dim=2, memory_shape=(10,10), threshold=1.0, actor_mode='reset'):
        """
        :param input_dim: размерность сенсорного входа
        :param memory_shape: форма памяти
        :param threshold: порог устойчивости для PhaseCore
        :param actor_mode: режим действия актора
        """
        self.sensor = Sensor(input_dim)
        self.memory = Memory(memory_shape)
        self.phase = PhaseCore(threshold)
        self.actor = Actor(actor_mode)

    def step(self):
        """
        Выполняет один шаг Difference Loop:
        1. Сенсор генерирует различие
        2. Память деформируется
        3. Фазовое ядро оценивает устойчивость
        4. Актор перестраивает память при необходимости
        :return: dict с результатами шага (различие, память, устойчивость, дисперсия)
        """
        diff = self.sensor.sense()
        self.memory.deform(diff)
        stable, variance = self.phase.check_stability(self.memory.get_state())
        stable = bool(stable)
        if not stable:
            self.actor.act(self.memory)
        return {
            'diff': diff,
            'memory': self.memory.get_state(),
            'stable': stable,
            'variance': variance
        }
