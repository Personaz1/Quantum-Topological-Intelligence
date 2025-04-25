import unittest
from QTI_Core.actor import Actor
from QTI_Core.memory import Memory
import numpy as np

class TestActor(unittest.TestCase):
    def test_act_reset(self):
        mem = Memory(shape=(3, 3))
        mem.state += 2.0
        actor = Actor(mode='reset')
        actor.act(mem)
        self.assertTrue(np.allclose(mem.state, 1.0))

    def test_act_amplify(self):
        mem = Memory(shape=(3, 3))
        mem.state += 2.0
        actor = Actor(mode='amplify')
        actor.act(mem)
        self.assertTrue(np.allclose(mem.state, 2.4))

if __name__ == '__main__':
    unittest.main()
