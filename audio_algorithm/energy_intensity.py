import numpy as np
from opensmile import Smile, FeatureSet, FeatureLevel

class EnergyIntensity:
    def __init__(self):
        self.smile = Smile(
            feature_set=FeatureSet.ComParE_2016,
            feature_level=FeatureLevel.LowLevelDescriptors
        )
        self.algorithm_map = {
            'pcm_RMSenergy': self.pcm_RMSenergy,
            'pcm_zcr': self.pcm_zcr,
            'intensity': self.intensity,
            'loudness': self.loudness
        }

    def __repr__(self):
        return f"{self.algorithm_map.keys()}"

    def extract(self, algorithm_name, audio_file):
        if algorithm_name not in self.algorithm_map:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        return self.algorithm_map[algorithm_name](audio_file)

    def pcm_RMSenergy(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['pcm_RMSenergy_sma']), np.std(result['pcm_RMSenergy_sma'])

    def pcm_zcr(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['pcm_zcr_sma']), np.std(result['pcm_zcr_sma'])

    def intensity(self, audio_file):
        result = self.smile.process_file(audio_file)
        return np.mean(result['audspec_lengthL1norm_sma']), np.std(result['audspec_lengthL1norm_sma'])

    def loudness(self, audio_file):
        # Note: There's no direct 'loudness' feature, so we'll use pcm_RMSenergy as a proxy
        result = self.smile.process_file(audio_file)
        return np.mean(result['pcm_RMSenergy_sma']), np.std(result['pcm_RMSenergy_sma'])