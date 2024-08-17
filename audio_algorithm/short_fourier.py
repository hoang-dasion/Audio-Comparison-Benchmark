# audio_algorithm/short_fourier.py

import librosa

class ShortFourier:
    def __init__(self):
        self.algorithm_map = {
            'mel': self.mel_spectrogram,
            'mfcc': self.mfcc,
            'chroma': self.chroma_feature,
            'spectral_contrast': self.spectral_contrast,
            'zcr': self.zero_crossing_rate,
            'spectral_rolloff': self.spectral_rolloff,
            'spectral_centroid': self.spectral_centroid,
            'rms': self.root_mean_square_energy
        }

    def extract(self, algorithm_name, y, sr):
        return self.algorithm_map[algorithm_name](y, sr)

    def mel_spectrogram(self, y, sr):
        return librosa.feature.melspectrogram(y=y, sr=sr)

    def mfcc(self, y, sr):
        return librosa.feature.mfcc(y=y, sr=sr)

    def chroma_feature(self, y, sr):
        return librosa.feature.chroma_stft(y=y, sr=sr)

    def spectral_contrast(self, y, sr):
        return librosa.feature.spectral_contrast(y=y, sr=sr)

    def zero_crossing_rate(self, y, sr):
        return librosa.feature.zero_crossing_rate(y)[0]

    def spectral_rolloff(self, y, sr):
        return librosa.feature.spectral_rolloff(y=y, sr=sr)

    def spectral_centroid(self, y, sr):
        return librosa.feature.spectral_centroid(y=y, sr=sr)

    def root_mean_square_energy(self, y, sr):
        return librosa.feature.rms(y=y)[0]