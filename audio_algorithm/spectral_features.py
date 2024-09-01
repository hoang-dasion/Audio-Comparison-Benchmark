import numpy as np
from opensmile import Smile, FeatureSet, FeatureLevel

class SpectralFeatures:
    def __init__(self):
        self.smile = Smile(
            feature_set=FeatureSet.ComParE_2016,
            feature_level=FeatureLevel.LowLevelDescriptors
        )
        self.algorithm_map = {
            'spectralFlux': self.spectral_flux,
            'spectralCentroid': self.spectral_centroid,
            'spectralEntropy': self.spectral_entropy,
            'spectralVariance': self.spectral_variance,
            'spectralSkewness': self.spectral_skewness,
            'spectralKurtosis': self.spectral_kurtosis
        }

    def __repr__(self):
        return f"{self.algorithm_map.keys()}"

    def extract(self, algorithm_name, audio_file):
        if algorithm_name not in self.algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        return self.algorithm_map[algorithm_name](audio_file)

    def spectral_flux(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['spectralFlux']), np.std(result['spectralFlux'])

    def spectral_centroid(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['spectralCentroid']), np.std(result['spectralCentroid'])

    def spectral_entropy(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['spectralEntropy']), np.std(result['spectralEntropy'])

    def spectral_variance(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['spectralVariance']), np.std(result['spectralVariance'])

    def spectral_skewness(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['spectralSkewness']), np.std(result['spectralSkewness'])

    def spectral_kurtosis(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['spectralKurtosis']), np.std(result['spectralKurtosis'])