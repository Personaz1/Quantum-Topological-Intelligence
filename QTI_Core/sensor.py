import librosa
import numpy as np

class Sensor:
    """
    Base sensor for the Difference Loop.
    Generates a stream of differences (e.g., noise).
    """
    def __init__(self, input_dim=2):
        """
        :param input_dim: dimension of the output difference vector
        """
        self.input_dim = input_dim

    def sense(self):
        """
        Generates a random difference vector (e.g., noise).
        :return: np.ndarray, shape=(input_dim,)
        """
        return np.random.randn(self.input_dim)

class AudioSensor(Sensor):
    """
    Audio sensor: converts an audio file into a stream of differences (MFCC vectors).
    Used to feed real sensor data into the Difference Loop.
    """
    def __init__(self, audio_path, input_dim=13, frame_length=2048, hop_length=512):
        """
        :param audio_path: path to the audio file
        :param input_dim: number of MFCC coefficients (dimension of differences)
        :param frame_length: STFT window length
        :param hop_length: hop length
        """
        super().__init__(input_dim)
        self.audio_path = audio_path
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.mfccs = None
        self._load_audio()

    def _load_audio(self):
        """
        Loads the audio file and computes MFCC vectors.
        """
        y, sr = librosa.load(self.audio_path, sr=None)
        # MFCCs: shape (n_mfcc, n_frames)
        self.mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.input_dim, n_fft=self.frame_length, hop_length=self.hop_length).T

    def sense_stream(self):
        """
        Generates a stream of differences (MFCC vectors) from the audio file.
        :yield: np.ndarray, shape=(input_dim,)
        """
        for mfcc in self.mfccs:
            yield mfcc

class WeightSensor(Sensor):
    """
    Weight sensor: converts neural network weights (or activations) into a stream of differences for the Difference Loop.
    Can be used to analyze the topological dynamics of modern models (PyTorch, TF).
    """
    def __init__(self, weights, input_dim=None):
        """
        :param weights: numpy array of weights or activations (any shape)
        :param input_dim: dimension of the output difference vector (default = weights dimension)
        """
        if input_dim is None:
            input_dim = weights.size if hasattr(weights, 'size') else len(weights)
        super().__init__(input_dim)
        self.weights = weights.flatten()
    def sense(self):
        """
        Returns weights as a difference vector (or their random projection if input_dim < weights dimension).
        :return: np.ndarray, shape=(input_dim,)
        """
        if self.input_dim == self.weights.shape[0]:
            return self.weights
        else:
            # Random projection of weights to input_dim space
            idx = np.random.choice(self.weights.shape[0], self.input_dim, replace=False)
            return self.weights[idx]

class MultiSensor(Sensor):
    """
    Multi-sensor: combines several sensors and aggregates their outputs.
    Can be used to feed multiple difference streams into the Difference Loop.
    """
    def __init__(self, sensors, mode='concat'):
        """
        :param sensors: list of sensor objects (Sensor, AudioSensor, WeightSensor, etc.)
        :param mode: aggregation method ('concat' - concatenation, 'sum' - sum)
        """
        self.sensors = sensors
        self.mode = mode
        input_dim = sum(s.input_dim for s in sensors) if mode == 'concat' else sensors[0].input_dim
        super().__init__(input_dim)

    def sense(self):
        """
        Aggregates the outputs of all sensors (concatenation by default).
        :return: np.ndarray, shape=(input_dim,)
        """
        outputs = [s.sense() for s in self.sensors]
        if self.mode == 'concat':
            return np.concatenate(outputs)
        elif self.mode == 'sum':
            return np.sum(outputs, axis=0)
        else:
            raise ValueError(f"Unknown aggregation mode: {self.mode}")
