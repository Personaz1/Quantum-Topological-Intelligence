from .sensor import Sensor
from .memory import Memory
from .phase import PhaseCore
from .actor import Actor

class QTI_Core:
    """
    Main Difference Loop cycle: S (Sensor) → M (Memory) → Φ (PhaseCore) → A (Actor) → S
    Encapsulates the entire QTI architecture and implements one step of the difference loop.
    """
    def __init__(self, sensor=None, input_dim=2, memory_shape=(10,10), threshold=1.0, actor_mode='reset', scheduler=None):
        """
        :param sensor: Sensor/MultiSensor/AudioSensor/WeightSensor object (default is basic Sensor)
        :param input_dim: sensor input dimension (if sensor is not provided)
        :param memory_shape: shape of memory
        :param threshold: stability threshold for PhaseCore
        :param actor_mode: actor's action mode
        :param scheduler: Scheduler object (Difference Loop scheduler)
        """
        if sensor is None:
            from .sensor import Sensor
        self.sensor = Sensor(input_dim)
        else:
            self.sensor = sensor
        self.memory = Memory(memory_shape)
        self.phase = PhaseCore(threshold)
        self.actor = Actor(actor_mode)
        self.scheduler = scheduler  # Placeholder for future scheduler

    def step(self):
        """
        Performs one step of the Difference Loop:
        1. Sensor generates a difference
        2. Memory is deformed
        3. PhaseCore evaluates stability
        4. Actor restructures memory if needed
        :return: dict with step results (difference, memory, stability, variance)
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
