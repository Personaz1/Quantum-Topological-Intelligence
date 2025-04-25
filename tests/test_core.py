import unittest
from QTI_Core.core import QTI_Core
import numpy as np

class TestQTICore(unittest.TestCase):
    def test_step(self):
        qti = QTI_Core(input_dim=2, memory_shape=(5, 5), threshold=0.1, actor_mode='reset')
        result = qti.step()
        self.assertIn('diff', result)
        self.assertIn('memory', result)
        self.assertIn('stable', result)
        self.assertIn('variance', result)
        self.assertEqual(result['diff'].shape, (2,))
        self.assertEqual(result['memory'].shape, (5, 5))
        self.assertIsInstance(result['stable'], bool)
        self.assertIsInstance(result['variance'], float)

if __name__ == '__main__':
    unittest.main()
