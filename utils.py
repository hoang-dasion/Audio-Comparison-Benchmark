# utils.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from const import AUDIO_ALGORITHMS, ML_ALGORITHMS, TARGET_COLUMNS, LABELS_FILE_NAMES, LABELS_OUTPUT_DIR
from datetime import datetime

def get_user_choices(args):
    choices = {
        'audio_algorithms': [],
        'feature_methods': {},
        'ml_algorithms': []
    }

    if args.custom:
        print("\nSelecting audio algorithms:")
        display_options(list(AUDIO_ALGORITHMS.keys()), "Available Audio Algorithms")
        choices['audio_algorithms'] = get_user_selection(list(AUDIO_ALGORITHMS.keys()), 
            "Enter the numbers of the audio algorithms you want to use (space-separated) or 'all': ")

        for algo in choices['audio_algorithms']:
            print(f"\nSelecting feature methods for {algo}:")
            display_options(AUDIO_ALGORITHMS[algo]['features'], f"Available Feature Extraction Methods for {algo}")
            choices['feature_methods'][algo] = get_user_selection(AUDIO_ALGORITHMS[algo]['features'], 
                f"Enter the numbers of the feature extraction methods you want to use for {algo} (space-separated) or 'all': ")

        print("\nSelecting ML algorithms:")
        display_options(list(ML_ALGORITHMS.keys()), "Available ML Algorithms")
        choices['ml_algorithms'] = get_user_selection(list(ML_ALGORITHMS.keys()), 
            "Enter the numbers of the ML algorithms you want to use (space-separated) or 'all': ")
    else:
        choices['audio_algorithms'] = list(AUDIO_ALGORITHMS.keys())
        for algo in choices['audio_algorithms']:
            choices['feature_methods'][algo] = AUDIO_ALGORITHMS[algo]['features']
        choices['ml_algorithms'] = list(ML_ALGORITHMS.keys())

    return choices

def process_algorithm(algorithm, args, audio_processor, feature_selector, ml_processor, selected_features, selected_ml_algorithms):
    print(f"\nProcessing algorithm: {algorithm}")
    
    feature_subsets = audio_processor.run_analysis(algorithm, args.audio_path, selected_features, force_reprocess=args.force_reprocess)

    if feature_subsets.empty:
        print(f"No features extracted for {algorithm}. Skipping this algorithm.")
        return None

    print(f"Loading and merging data for {algorithm}...")
    merged_data = load_and_merge_data(LABELS_FILE_NAMES, feature_subsets)

    print(f"Splitting data for {algorithm}...")
    X = merged_data.drop(['Participant_ID'] + TARGET_COLUMNS, axis=1)
    y = merged_data[TARGET_COLUMNS]

    X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.2, random_state=42)

    print("Performing unsupervised feature selection...")
    selected_feature_names, X_selected = feature_selector.unsupervised_feature_selection(X, X.columns)

    X_train_selected = X_train[selected_feature_names]
    X_dev_selected = X_dev[selected_feature_names]
    X_test_selected = X_test[selected_feature_names]

    algorithm_results = {}

    for target in TARGET_COLUMNS:
        print(f"Running ML models for target: {target}")
        y_train_target = y_train[target]
        y_dev_target = y_dev[target]
        y_test_target = y_test[target]

        ml_results = ml_processor.run_ml_pipeline(
            X_train_selected, y_train_target, X_dev_selected, y_dev_target, X_test_selected, y_test_target, 
            selected_ml_algorithms, algorithm=algorithm, sub_algorithm=target, target=target, use_cache=args.use_cache
        )

        algorithm_results[target] = {feature: ml_results for feature in selected_feature_names}

    return algorithm_results

def create_time_based_output_directory(base_path="./output"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_dir = os.path.join(base_path, current_time)
    os.makedirs(os.path.join(time_dir, "cached_models"), exist_ok=True)
    os.makedirs(os.path.join(time_dir, "plots"), exist_ok=True)
    return time_dir

def check_audio_files(audio_path):
    if not os.path.exists(audio_path):
        raise ValueError(f"The specified audio path does not exist: {audio_path}")
    wav_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    if not wav_files:
        raise ValueError(f"No .wav files found in the specified audio path: {audio_path}")
    return wav_files

def check_label_files():
    for file in LABELS_FILE_NAMES:
        file_path = os.path.join(LABELS_OUTPUT_DIR, f"complete_{file}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Label file not found: {file_path}")

def load_and_merge_data(label_files, feature_subsets):
    all_labels = pd.concat([pd.read_csv(os.path.join(LABELS_OUTPUT_DIR, f"complete_{file}.csv")) for file in label_files])
    all_labels['Participant_ID'] = all_labels['Participant_ID'].astype(str)
    feature_subsets['Participant_ID'] = feature_subsets['Participant_ID'].astype(str)
    
    merged_data = pd.merge(feature_subsets, all_labels, on='Participant_ID', how='inner')
        
    if merged_data.shape[0] == 0:
        print("Warning: No data after merging. Checking for mismatched Participant_IDs...")
        label_ids = set(all_labels['Participant_ID'])
        feature_ids = set(feature_subsets['Participant_ID'])
        print(f"Label IDs not in features: {label_ids - feature_ids}")
        print(f"Feature IDs not in labels: {feature_ids - label_ids}")
        raise ValueError("No data available after merging features and labels.")
    
    return merged_data

def display_options(options, title):
    print(f"\n{title}:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print(f"{len(options) + 1}. All")

def get_user_selection(options, prompt):
    while True:
        try:
            user_input = input(prompt).strip().lower()
            if user_input == 'all':
                return options
            selections = [int(x) for x in user_input.split()]
            if all(1 <= s <= len(options) + 1 for s in selections):
                if len(options) + 1 in selections:
                    return options
                return [options[i - 1] for i in selections]
            else:
                raise ValueError
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces or 'all'.")
