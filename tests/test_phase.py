import unittest
from QTI_Core.phase import PhaseCore
import numpy as np

class TestPhaseCore(unittest.TestCase):
    def test_check_stability(self):
        phase = PhaseCore(threshold=0.1)
        stable_state = np.zeros((5, 5))
        unstable_state = np.random.randn(5, 5) * 10
        stable, var1 = phase.check_stability(stable_state)
        unstable, var2 = phase.check_stability(unstable_state)
        self.assertTrue(stable)
        self.assertFalse(unstable)

if __name__ == '__main__':
    unittest.main()
