import os
import numpy as np
import pandas as pd
import librosa
from glob import glob
from tqdm import tqdm
import itertools
from const import AUDIO_ALGORITHMS
import importlib
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

class AudioProcessor:
    def __init__(self):
        self.algorithms = self._load_algorithms()
        self.output_dir = "./output/m4a/features"

    def _load_algorithms(self):
        algorithms = {}
        for algo_name, algo_info in AUDIO_ALGORITHMS.items():
            module = importlib.import_module(f"audio_algorithm.{algo_info['file'][:-3]}")
            algo_class = getattr(module, algo_info['class'])
            algorithms[algo_name] = algo_class()
        return algorithms

    def run_analysis(self, audio_path, file_extension="m4a", force_reprocess=False):
        all_features = []

        for algorithm, algo_info in AUDIO_ALGORITHMS.items():
            feature_names = algo_info['features']
            output_file = os.path.join(self.output_dir, f"{algorithm}_features.csv")

            if os.path.exists(output_file) and not force_reprocess:
                print(f"Loading existing features for {algorithm} from {output_file}")
                features = pd.read_csv(output_file)
            else:
                print(f"Processing audio files for algorithm: {algorithm}")
                features = self.process_audio_files(algorithm, feature_names, audio_path, file_extension)
                if features.empty:
                    print(f"No features extracted for {algorithm}. Skipping feature combination creation.")
                    continue
                print("Creating feature combinations...")
                features = self.create_feature_combinations(features, feature_names)
                os.makedirs(self.output_dir, exist_ok=True)
                features.to_csv(output_file, index=False)
                print(f"Saved features for {algorithm} to {output_file}")

            print(f"Extracted features shape for {algorithm}: {features.shape}")
            print(f"Unique Participant_IDs in features for {algorithm}: {features['Participant_ID'].nunique()}")
            
            all_features.append(features)
        
        # Combine all features into a single DataFrame
        combined_features = self.combine_all_features(all_features)
        
        print(f"Final combined features shape: {combined_features.shape}")
        print(f"Unique Participant_IDs in final features: {combined_features['Participant_ID'].nunique()}")
        
        # Save the combined features to a single CSV file
        combined_output_file = os.path.join(self.output_dir, "all_features_combined.csv")
        combined_features.to_csv(combined_output_file, index=False)
        print(f"Saved combined features to {combined_output_file}")
        
        return combined_features

    def process_audio_files(self, algorithm, feature_names, audio_path, file_extension):
        if os.path.isdir(audio_path):
            audio_files = glob(os.path.join(audio_path, f"*.{file_extension}"))
        elif os.path.isfile(audio_path):
            audio_files = [audio_path]
        else:
            print(f"Invalid audio path: {audio_path}")
            return pd.DataFrame()

        all_features = []
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            features = self.process_single_audio_file(algorithm, feature_names, audio_file)
            if features is not None:
                all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        columns = ['Participant_ID'] + feature_names
        return pd.DataFrame(all_features, columns=columns)

    def process_single_audio_file(self, algorithm, feature_names, audio_file):
        try:
            participant_id = os.path.basename(audio_file).split('.')[0]
            y, sr = librosa.load(audio_file)
            features = [participant_id]

            for feature_name in feature_names:
                if algorithm == 'short_fourier':
                    feature_value = self.algorithms[algorithm].extract(feature_name, y, sr)
                elif algorithm in ['energy_intensity', 'spectral_features', 'voice_quality']:
                    feature_value = self.algorithms[algorithm].extract(feature_name, audio_file)
                else:
                    feature_value = self.algorithms[algorithm].extract(feature_name, y)
                
                if isinstance(feature_value, (tuple, np.ndarray)):
                    feature_mean = np.mean(feature_value)
                else:
                    feature_mean = feature_value
                
                features.append(feature_mean)

            return features
        except Exception as e:
            print(f"Error processing file {audio_file} for algorithm {algorithm}: {str(e)}")
            return None

    def create_feature_combinations(self, individual_features, feature_names):
        all_combinations = []
        for r in range(2, len(feature_names) + 1):
            all_combinations.extend(itertools.combinations(feature_names, r))

        new_features = {}
        for combination in tqdm(all_combinations, desc="Creating feature combinations"):
            column_name = f"({', '.join(combination)})"
            new_features[column_name] = individual_features[list(combination)].prod(axis=1)

        combined_features = pd.concat([individual_features, pd.DataFrame(new_features)], axis=1)
        return combined_features

    def combine_all_features(self, feature_dataframes):
        combined = feature_dataframes[0]
        for df in feature_dataframes[1:]:
            combined = pd.merge(combined, df, on='Participant_ID', how='outer', suffixes=('', '_drop'))
        
        # Remove any columns with '_drop' suffix
        combined = combined[[col for col in combined.columns if not col.endswith('_drop')]]
        
        return combined

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process audio files and generate a CSV file.")
    parser.add_argument("data_path", help="Path to the folder containing the data files")
    parser.add_argument("file_extension", help="File extension of the audio files to process")
    args = parser.parse_args()

    processor = AudioProcessor()
    processor.run_analysis(args.data_path, args.file_extension)

if __name__ == "__main__":
    main()