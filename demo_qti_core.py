from QTI_Core.core import QTI_Core
from QTI_Core.visualize import plot_memory
import time
import numpy as np
from QTI_Core.sensor import Sensor
from QTI_Core.memory import Memory

if __name__ == '__main__':
    qti = QTI_Core(input_dim=2, memory_shape=(20, 20), threshold=0.5, actor_mode='reset')
    steps = 100
    for i in range(steps):
        result = qti.step()
        if i % 10 == 0:
            print(f"Step {i}: variance={result['variance']:.4f}, stable={result['stable']}")
            plot_memory(result['memory'], title=f"Memory State at step {i}")
        time.sleep(0.1)

    sensor = Sensor(input_dim=2)
    memory = Memory(shape=(10, 10))
    steps = 30
    for _ in range(steps):
        diff = sensor.sense()
        memory.deform(diff)
    print("Визуализация фазовых переходов...")
    memory.plot_phases(method="ripser") 