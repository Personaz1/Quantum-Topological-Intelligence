import librosa
import numpy as np

class Sensor:
    """
    Базовый сенсор для Difference Loop.
    Генерирует поток различий (например, шум).
    """
    def __init__(self, input_dim=2):
        """
        :param input_dim: размерность выходного вектора различий
        """
        self.input_dim = input_dim

    def sense(self):
        """
        Генерирует случайный вектор различий (например, шум).
        :return: np.ndarray, shape=(input_dim,)
        """
        return np.random.randn(self.input_dim)

class AudioSensor(Sensor):
    """
    Аудиосенсор: преобразует аудиофайл в поток различий (MFCC-векторы).
    Используется для подачи реальных сенсорных данных в Difference Loop.
    """
    def __init__(self, audio_path, input_dim=13, frame_length=2048, hop_length=512):
        """
        :param audio_path: путь к аудиофайлу
        :param input_dim: число MFCC-коэффициентов (размерность различий)
        :param frame_length: длина окна для STFT
        :param hop_length: шаг окна
        """
        super().__init__(input_dim)
        self.audio_path = audio_path
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.mfccs = None
        self._load_audio()

    def _load_audio(self):
        """
        Загружает аудиофайл и вычисляет MFCC-векторы.
        """
        y, sr = librosa.load(self.audio_path, sr=None)
        # MFCCs: shape (n_mfcc, n_frames)
        self.mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.input_dim, n_fft=self.frame_length, hop_length=self.hop_length).T

    def sense_stream(self):
        """
        Генерирует поток различий (MFCC-векторы) из аудиофайла.
        :yield: np.ndarray, shape=(input_dim,)
        """
        for mfcc in self.mfccs:
            yield mfcc
