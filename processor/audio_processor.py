import os
import sys
import time
import json
import csv
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import stats
from tqdm import tqdm

from audio_algorithm.time_domain import TimeDomain
from audio_algorithm.fourier import Fourier
from audio_algorithm.short_fourier import ShortFourier
from audio_algorithm.wavelet import Wavelet
from const import FEATURES_DIC, METRICS_DIC
from plot.audio_plot import AudioPlot
from utils import save_params, save_config

class AudioProcessor:
    def __init__(self):
        self.time_domain = TimeDomain()
        self.fourier = Fourier()
        self.short_fourier = ShortFourier()
        self.wavelet = Wavelet()

    def load_params(self, params_file):
        default_params = {'sr': 22050, 'n_components': 40}
        if params_file:
            params_file = params_file if params_file.endswith('.json') else f"{params_file}.json"
            if os.path.exists(params_file):
                try:
                    with open(params_file, 'r') as f:
                        custom_params = json.load(f)
                        default_params.update(custom_params)
                except json.JSONDecodeError as e:
                    print(f"Error loading parameter file: {e}")
        return default_params

    def compact_feature_representation(self, features, n_components=40):
        def process_array(arr):
            if arr.ndim == 1:
                return np.hstack([
                    np.mean(arr),
                    np.std(arr),
                    stats.skew(arr),
                    stats.kurtosis(arr)
                ])
            else:
                return np.hstack([
                    np.mean(arr, axis=1),
                    np.std(arr, axis=1),
                    stats.skew(arr, axis=1),
                    stats.kurtosis(arr, axis=1)
                ])

        if isinstance(features, np.ndarray):
            return process_array(features)[:n_components]
        elif isinstance(features, list):
            processed = []
            for item in features:
                if isinstance(item, np.ndarray):
                    processed.extend(process_array(item))
                elif isinstance(item, list):
                    processed.extend(self.compact_feature_representation(item, n_components))
                else:
                    processed.append(item)
            return np.array(processed)[:n_components]
        else:
            return np.array([features])[:n_components]

    def process_audio_file(self, file_path, durations, feature_names, params, output_dir, algorithm):
        results = {}
        sr = params.get('sr', 22050)
        n_components = params.get('n_components', 40)
        
        for feature_name in feature_names:
            results[feature_name] = {}
            for duration in durations:
                start_time = time.time()
                
                if duration == 'full':
                    y, sr = librosa.load(file_path, sr=sr)

                else:
                    y, sr = librosa.load(file_path, duration=float(duration), sr=sr)
                
                if algorithm == 'time_domain':
                    features = self.time_domain.extract(feature_name, y)
                elif algorithm == 'fourier':
                    features = self.fourier.extract(feature_name, y)
                elif algorithm == 'short_fourier':
                    features = self.short_fourier.extract(feature_name, y, sr)
                elif algorithm == 'wavelet':
                    features = self.wavelet.extract(feature_name, y)
                
                compact_features = self.compact_feature_representation(features, n_components)
                
                end_time = time.time()
                
                computation_time = (end_time - start_time) * 1000  # Convert to milliseconds
                memory_usage = sys.getsizeof(compact_features) / (1024 * 1024)  # Convert to MB
                
                # Extract label from filename
                filename = os.path.basename(file_path)
                label = int(filename.split('-')[2]) - 1  # Assuming the third part of the filename indicates the label
                
                results[feature_name][duration] = {
                    'time': computation_time,
                    'memory': memory_usage,
                    'dimensionality': compact_features.shape,
                    'features': compact_features,
                    'label': label
                }

        return results
    
    def run_analysis(self, algorithm, runtimes=1, durations="", feature_names=None, metrics=None, 
                     output_dir='output', params_file='params', audio_path=None):
        # Initialize parameters
        feature_names = feature_names or FEATURES_DIC[algorithm]
        metrics = metrics or list(METRICS_DIC.keys())
        durations = durations.split(',') if durations else ["full"]

        # Load and save configurations
        params = self.load_params(params_file)
        save_params(params, params_file, output_dir, algorithm)
        save_config(algorithm, runtimes, durations, feature_names, params, metrics, output_dir, audio_path)

        # Initialize results storage
        results = {feature: {duration: {} for duration in durations} for feature in feature_names}
        combined_features = {feature: [] for feature in feature_names}
        combined_labels = []

        # Extract parameters
        sr = params.get('sr', 22050)
        n_components = params.get('n_components', 40)

        # Process audio data
        if audio_path is None:
            self._get_random_data(algorithm, runtimes, durations, feature_names, sr, n_components, 
                                      results, combined_features, combined_labels, output_dir)
        else:
            self._get_audio_files(audio_path, algorithm, durations, feature_names, params, 
                                      results, combined_features, combined_labels, output_dir)

        # Create and save feature arrays
        feature_arrays = self._create_feature_arrays(feature_names, combined_features, output_dir, algorithm)

        # Save labels
        self._save_labels(combined_labels, output_dir, algorithm)

        # Plot metric comparisons if necessary
        if len(durations) > 1 or (len(durations) == 1 and durations[0] != 'full'):
            self._plot_metric_comparisons(results, metrics, output_dir, algorithm)

        return feature_arrays, np.array(combined_labels)

    def _get_random_data(self, algorithm, runtimes, durations, feature_names, sr, n_components, 
                             results, combined_features, combined_labels, output_dir):
        print(f"No audio path provided. Generating random data for {algorithm} analysis.")
        for i in tqdm(range(runtimes), desc="Generating random data"):
            for duration in durations:
                dur = 10 if duration == 'full' else float(duration)
                y = np.random.randn(int(dur * sr))
                for feature_name in feature_names:
                    features = self._extract_features(algorithm, feature_name, y, sr)
                    compact_features = self.compact_feature_representation(features, n_components)
                    results[feature_name][duration][f'generated_data_{i}'] = {
                        'time': 0,
                        'memory': sys.getsizeof(compact_features) / (1024 * 1024),
                        'dimensionality': compact_features.shape,
                        'features': compact_features,
                        'label': np.random.randint(0, 8)
                    }
                    combined_features[feature_name].append(compact_features)
                    combined_labels.append(results[feature_name][duration][f'generated_data_{i}']['label'])
                    AudioPlot.plot_feature(features, algorithm, feature_name, duration, output_dir, sr)

    def _get_audio_files(self, audio_path, algorithm, durations, feature_names, params, 
                             results, combined_features, combined_labels, output_dir):
        wav_files = [audio_path] if os.path.isfile(audio_path) else glob(os.path.join(audio_path, '*.wav'))
        if not wav_files:
            raise ValueError("No WAV files found in the specified path.")

        for wav_file in tqdm(wav_files, desc="Processing audio files"):
            file_name = os.path.basename(wav_file)
            label = int(file_name[7:8]) - 1  # Assume file format like 03-01-01-01-01-01-01.wav
            combined_labels.append(label)
            
            file_results = self.process_audio_file(wav_file, durations, feature_names, params, output_dir, algorithm)
            
            for feature_name in feature_names:
                feature_vector = []
                for duration in durations:
                    compact_features = file_results[feature_name][duration]['features']
                    results[feature_name][duration][file_name] = file_results[feature_name][duration]
                    feature_vector.extend(compact_features)
                combined_features[feature_name].append(np.array(feature_vector))

    def _extract_features(self, algorithm, feature_name, y, sr):
        if algorithm == 'time_domain':
            return self.time_domain.extract(feature_name, y)
        elif algorithm == 'fourier':
            return self.fourier.extract(feature_name, y)
        elif algorithm == 'short_fourier':
            return self.short_fourier.extract(feature_name, y, sr)
        elif algorithm == 'wavelet':
            return self.wavelet.extract(feature_name, y)

    def _create_feature_arrays(self, feature_names, combined_features, output_dir, algorithm):
        feature_arrays = {}
        for feature_name in feature_names:
            feature_arrays[feature_name] = np.array(combined_features[feature_name])
            
            csv_dir = os.path.join(output_dir, algorithm, feature_name, 'data')
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f'{feature_name}_all_data.csv')
            np.savetxt(csv_path, feature_arrays[feature_name], delimiter=',')
            
            print(f"Saved to: {csv_path}")
        
        return feature_arrays

    def _save_labels(self, combined_labels, output_dir, algorithm):
        labels_path = os.path.join(output_dir, algorithm, 'labels.csv')
        np.savetxt(labels_path, np.array(combined_labels), delimiter=',', fmt='%d')
        print(f"Labels saved to: {labels_path}")
        print("Done!")

    def _plot_metric_comparisons(self, results, metrics, output_dir, algorithm):
        for metric in metrics:
            AudioPlot.plot_metric_comparison(results, metric, output_dir, algorithm) 