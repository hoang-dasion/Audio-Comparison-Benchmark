import numpy as np
from opensmile import Smile, FeatureSet, FeatureLevel

class VoiceQuality:
    def __init__(self):
        self.smile = Smile(
            feature_set=FeatureSet.ComParE_2016,
            feature_level=FeatureLevel.LowLevelDescriptors
        )
        self.algorithm_map = {
            'voiceProb': self.voice_prob,
            'F0': self.fundamental_frequency,
            'jitter': self.jitter,
            'shimmer': self.shimmer
        }

    def __repr__(self):
        return f"{self.algorithm_map.keys()}"

    def extract(self, algorithm_name, audio_file):
        if algorithm_name not in self.algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        return self.algorithm_map[algorithm_name](audio_file)

    def voice_prob(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['voiceProb']), np.std(result['voiceProb'])

    def fundamental_frequency(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['F0final']), np.std(result['F0final'])

    def jitter(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['jitterLocal']), np.std(result['jitterLocal'])

    def shimmer(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['shimmerLocal']), np.std(result['shimmerLocal'])