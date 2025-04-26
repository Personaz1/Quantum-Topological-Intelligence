import unittest
from QTI_Core.sensor import Sensor, AudioSensor
import numpy as np
import soundfile as sf
import os

class TestSensor(unittest.TestCase):
    def test_sense_output(self):
        sensor = Sensor(input_dim=3)
        diff = sensor.sense()
        self.assertEqual(diff.shape, (3,))
        self.assertIsInstance(diff, np.ndarray)

    def test_audio_sensor(self):
        # Generate a short synthetic audio file
        sr = 16000
        y = np.random.randn(sr) * 0.01
        fname = "test_audio.wav"
        sf.write(fname, y, sr)
        try:
            audio_sensor = AudioSensor(fname, input_dim=5)
            stream = list(audio_sensor.sense_stream())
            self.assertTrue(len(stream) > 0)
            self.assertEqual(stream[0].shape, (5,))
        finally:
            os.remove(fname)

if __name__ == '__main__':
    unittest.main()
