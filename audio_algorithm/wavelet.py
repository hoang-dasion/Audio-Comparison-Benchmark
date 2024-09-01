import numpy as np
import pywt

class Wavelet:
    def __init__(self):
        self.algorithm_map = {
            'wavelet_coefficients': self.wavelet_coefficients,
            'wavelet_packet': self.wavelet_packet,
            'mra': self.multi_resolution_analysis,
            'wavelet_energy': self.wavelet_energy,
            'wavelet_entropy': self.wavelet_entropy
        }
        self.wavelet = 'db4'
        self.level = 5

    def __repr__(self):
        return f"{self.algorithm_map.keys()}"

    def extract(self, algorithm_name, y):
        if algorithm_name not in self.algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        return self.algorithm_map[algorithm_name](y)

    def trim_coeffs(self, coeffs, target_length):
        trimmed = []
        for coeff in coeffs:
            if len(coeff) > target_length:
                trim_amount = len(coeff) - target_length
                if trim_amount % 2 == 0:
                    trimmed.append(coeff[trim_amount//2 : -trim_amount//2])
                else:
                    trimmed.append(coeff[trim_amount//2 : -(trim_amount//2 + 1)])
            else:
                trimmed.append(coeff)
        return trimmed

    def wavelet_coefficients(self, y):
        coeffs = pywt.wavedec(y, self.wavelet, level=self.level)
        trimmed_coeffs = self.trim_coeffs(coeffs, len(y))
        return np.concatenate(trimmed_coeffs)

    def wavelet_packet(self, y):
        wp = pywt.WaveletPacket(data=y, wavelet=self.wavelet, mode='symmetric')
        coeffs = [node.data for node in wp.get_level(self.level, 'natural')]
        trimmed_coeffs = self.trim_coeffs(coeffs, len(y))
        return np.concatenate(trimmed_coeffs)

    def multi_resolution_analysis(self, y):
        coeffs = pywt.wavedec(y, self.wavelet, level=self.level)
        trimmed_coeffs = self.trim_coeffs(coeffs, len(y))
        reconstructed = [pywt.waverec(trimmed_coeffs[:i+1] + [None]*(len(trimmed_coeffs)-i-1), self.wavelet)[:len(y)] for i in range(len(trimmed_coeffs))]
        return np.concatenate(reconstructed)

    def wavelet_energy(self, y):
        coeffs = pywt.wavedec(y, self.wavelet, level=self.level)
        trimmed_coeffs = self.trim_coeffs(coeffs, len(y))
        return np.array([np.sum(np.square(c)) for c in trimmed_coeffs])

    def wavelet_entropy(self, y):
        coeffs = pywt.wavedec(y, self.wavelet, level=self.level)
        trimmed_coeffs = self.trim_coeffs(coeffs, len(y))
        total_energy = np.sum([np.sum(np.square(c)) for c in trimmed_coeffs])
        probabilities = [np.sum(np.square(c))/total_energy for c in trimmed_coeffs if np.sum(np.square(c)) > 0]
        return np.array([-np.sum(p * np.log2(p) for p in probabilities)])