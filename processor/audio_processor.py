# audio_processor.py

import os
import numpy as np
import pandas as pd
import librosa
from glob import glob
from tqdm import tqdm
import itertools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from const import AUDIO_ALGORITHMS
import importlib

class AudioProcessor:
    def __init__(self):
        self.algorithms = self._load_algorithms()
        self.output_dir = "./output/features"
        self.lock = threading.Lock()

    def _load_algorithms(self):
        algorithms = {}
        for algo_name, algo_info in AUDIO_ALGORITHMS.items():
            module = importlib.import_module(f"audio_algorithm.{algo_info['file'][:-3]}")
            algo_class = getattr(module, algo_info['class'])
            algorithms[algo_name] = algo_class()
        return algorithms

    def run_analysis(self, algorithm, audio_path, force_reprocess=False):
        feature_names = AUDIO_ALGORITHMS[algorithm]['features']
        output_file = os.path.join(self.output_dir, f"{algorithm}_features.csv")

        if os.path.exists(output_file) and not force_reprocess:
            print(f"Loading existing features for {algorithm} from {output_file}")
            features = pd.read_csv(output_file)
        else:
            print(f"Processing audio files for algorithm: {algorithm}")
            individual_features = self.process_audio_files(algorithm, feature_names, audio_path)
            if individual_features.empty:
                print(f"No features extracted for {algorithm}. Skipping feature combination creation.")
                return pd.DataFrame()
            print("Creating feature combinations...")
            features = self.create_all_feature_combinations(individual_features, feature_names)
            os.makedirs(self.output_dir, exist_ok=True)
            features.to_csv(output_file, index=False)
            print(f"Saved all features to {output_file}")

        print(f"Extracted features shape: {features.shape}")
        print(f"Unique Participant_IDs in features: {features['Participant_ID'].nunique()}")
        return features

    def process_audio_files(self, algorithm, feature_names, audio_path):
        if os.path.isdir(audio_path):
            audio_files = glob(os.path.join(audio_path, "*.wav"))
        elif os.path.isfile(audio_path):
            audio_files = [audio_path]
        else:
            print(f"Invalid audio path: {audio_path}")
            return pd.DataFrame()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_single_audio_file, algorithm, feature_names, wav_file) 
                       for wav_file in audio_files]
            all_features = [future.result() for future in tqdm(as_completed(futures), total=len(audio_files), desc="Processing audio files")]

        all_features = [f for f in all_features if f is not None]
        if not all_features:
            return pd.DataFrame()

        columns = ['Participant_ID'] + [f"{name}_{stat}" for name in feature_names for stat in ['mean', 'std']]
        return pd.DataFrame(all_features, columns=columns)

    def process_single_audio_file(self, algorithm, feature_names, wav_file):
        try:
            participant_id = os.path.basename(wav_file).split('.')[0]
            features = [participant_id]

            for feature_name in feature_names:
                feature_mean, feature_std = self.extract_feature(algorithm, feature_name, wav_file)
                features.extend([feature_mean, feature_std])

            return features
        except Exception as e:
            print(f"Error processing file {wav_file}: {str(e)}")
            return None

    def extract_feature(self, algorithm, feature_name, audio_file):
        return self.algorithms[algorithm].extract(feature_name, audio_file)

    def create_all_feature_combinations(self, individual_features, feature_names):
        all_combinations = []
        for r in range(1, len(feature_names) + 1):
            all_combinations.extend(itertools.combinations(feature_names, r))

        new_features = {}
        for combination in tqdm(all_combinations, desc="Creating feature combinations"):
            if len(combination) > 1:
                column_name = f"({', '.join(combination)})"
                new_features[column_name] = individual_features[[f"{name}_mean" for name in combination]].prod(axis=1)

        # Combine original features with new combinations
        combined_features = pd.concat([individual_features, pd.DataFrame(new_features)], axis=1)
        return combined_features