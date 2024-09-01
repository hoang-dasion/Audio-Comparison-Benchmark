import os
import pandas as pd
from pydub import AudioSegment
import shutil
from .daic_woz_prep import process_rejoined_files, process_rejoined_files_spec

import numpy as np
import librosa
from scipy import stats
from tqdm import tqdm

class AudioProcessorOscar:
    def __init__(self):
        pass
    
    def process_audio(self, audio_path, output_dir):
        self.process_all_files(audio_path, os.path.join(output_dir, 'cut'), calc_segments=True)
        self.rejoin_segments(os.path.join(output_dir, 'cut', 'all'), os.path.join(output_dir, 'joined'))
        process_rejoined_files(os.path.join(output_dir, 'joined'), os.path.join(output_dir, 'os'))
        process_rejoined_files_spec(os.path.join(output_dir, 'joined'), os.path.join(output_dir, 'spec'))

    def load_features(self, os_dir):
        features = []
        file_names = []

        for file in os.listdir(os_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(os_dir, file))
                features.append(df.values)
                file_names.append(file.split('.')[0])  # Store file name without extension

        return np.vstack(features), np.array(file_names)

    def run_analysis(self, audio_path, output_dir):
        self.process_audio(audio_path, output_dir)
        features, file_names = self.load_features(os.path.join(output_dir, 'os'))
        return features, file_names

    # Over-ride from daic_woz_prep to deal without labels.csv
    def process_all_files(self, in_path, out_path_audio, calc_segments=True):
        subdir_main = os.path.join(out_path_audio, 'all')
        os.makedirs(subdir_main, exist_ok=True)

        if calc_segments:
            for file in tqdm(os.listdir(in_path)):
                if file.endswith('.wav'):
                    filename = file.split('.')[0]
                    print(f'Processing {filename}')
                    file_dir = os.path.join(subdir_main, filename)
                    os.makedirs(file_dir, exist_ok=True)
                    self.process_file(in_path, file_dir, filename)

    def process_file(self, in_path, out_path_audio, name, min_length_ms=500):
        audio_path = os.path.join(in_path, name + '.wav')

        audio = AudioSegment.from_wav(audio_path)

        segment_length_ms = 5000  # 5 seconds
        for i, start in enumerate(range(0, len(audio), segment_length_ms)):
            end = start + segment_length_ms
            if end - start >= min_length_ms:
                segment = audio[start:end]
                segment_path_i = os.path.join(out_path_audio, f'{name}_segment_{i}.wav')
                segment.export(segment_path_i, format='wav')

    def rejoin_segments(self, in_path, out_path):
        os.makedirs(out_path, exist_ok=True)

        for participant_dir in os.listdir(in_path):
            participant_path = os.path.join(in_path, participant_dir)
            if os.path.isdir(participant_path):
                joined = AudioSegment.empty()
                for segment in sorted(os.listdir(participant_path)):
                    if segment.endswith('.wav'):
                        segment_path = os.path.join(participant_path, segment)
                        segment_audio = AudioSegment.from_wav(segment_path)
                        joined += segment_audio

                joined.export(os.path.join(out_path, f'{participant_dir}.wav'), format='wav')