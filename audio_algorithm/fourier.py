# audio_algorithm/fourier.py

import numpy as np

class Fourier:
    def __init__(self):
        self.algorithm_map = {
            'frequency_spectrum': self.frequency_spectrum,
            'cepstrum': self.cepstrum,
            'magnitude_spectrum': self.magnitude_spectrum,
            'phase_spectrum': self.phase_spectrum
        }

    def extract(self, algorithm_name, y):
        return self.algorithm_map[algorithm_name](y)

    def frequency_spectrum(self, y):
        return np.abs(np.fft.fft(y))

    def cepstrum(self, y):
        return np.fft.ifft(np.log(np.abs(np.fft.fft(y)))).real

    def magnitude_spectrum(self, y):
        return np.abs(np.fft.fft(y))

    def phase_spectrum(self, y):
        return np.angle(np.fft.fft(y))