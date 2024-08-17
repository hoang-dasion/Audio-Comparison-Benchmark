import numpy as np
import librosa
from scipy import signal, stats

class TimeDomain:
    def __init__(self):
        self.algorithm_map = {
            'zcr': self.zero_crossing_rate,
            'rms': self.root_mean_square_energy,
            'envelope': self.envelope,
            'temporal_moments': self.temporal_moments,
            'peak_to_peak': self.peak_to_peak_amplitude
        }

    def __repr__(self):
        print(f"{self.algorithm.map.keys()}")        

    def extract(self, algorithm_name, y):
        return self.algorithm_map[algorithm_name](y)

    def zero_crossing_rate(self, y):
        return librosa.feature.zero_crossing_rate(y)[0]

    def root_mean_square_energy(self, y):
        return librosa.feature.rms(y=y)[0]

    def envelope(self, y):
        analytic_signal = signal.hilbert(y)
        return np.abs(analytic_signal)

    def temporal_moments(self, y):
        mean = np.mean(y)
        variance = np.var(y)
        skewness = stats.skew(y)
        kurtosis = stats.kurtosis(y)
        return np.array([mean, variance, skewness, kurtosis])

    def peak_to_peak_amplitude(self, y):
        return np.max(y) - np.min(y)