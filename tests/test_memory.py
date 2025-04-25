import unittest
from QTI_Core.memory import Memory
import numpy as np

class TestMemory(unittest.TestCase):
    def test_init_and_deform(self):
        mem = Memory(shape=(5, 5))
        self.assertEqual(mem.get_state().shape, (5, 5))
        before = mem.get_state().copy()
        diff = np.array([1.0, 2.0])
        mem.deform(diff)
        after = mem.get_state()
        self.assertTrue(np.any(after != before))

    def test_persistent_homology(self):
        mem = Memory(shape=(5, 5))
        # Добавим несколько различий
        for _ in range(10):
            diff = np.random.randn(2)
            mem.deform(diff)
        dgms = mem.persistent_homology(method="ripser")
        self.assertIsNotNone(dgms)
        self.assertTrue(len(dgms) > 0)

if __name__ == '__main__':
    unittest.main()
