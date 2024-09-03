import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from processor.audio_processor import AudioProcessor
from processor.feature_selection_processor import FeatureSelectionProcessor
from processor.ml_processor import MLProcessor
from plot.ml_plot import MLPlot
from const import AUDIO_ALGORITHMS, ML_ALGORITHMS, TARGET_COLUMNS, LABELS_FILE_NAMES, LABELS_INPUT_DIR, LABELS_OUTPUT_DIR
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def main(args):
    print("Checking audio files...")
    wav_files = check_audio_files(args.audio_path)
    print(f"Found {len(wav_files)} .wav files.")

    print("Checking label files...")
    check_label_files()

    print("Initializing processors...")
    audio_processor = AudioProcessor()
    feature_selector = FeatureSelectionProcessor()
    ml_processor = MLProcessor(args.ml_output_dir)

    all_results = {target: {} for target in TARGET_COLUMNS}

    # User selection for audio algorithms
    if args.audio_algorithms:
        display_options(list(AUDIO_ALGORITHMS.keys()), "Available Audio Algorithms")
        algorithms_to_process = get_user_selection(list(AUDIO_ALGORITHMS.keys()), "Enter the numbers of the audio algorithms you want to use (space-separated) or 'all': ")
    else:
        algorithms_to_process = list(AUDIO_ALGORITHMS.keys())
        print("Using all available audio algorithms.")

    for algorithm in algorithms_to_process:
        print(f"\nProcessing algorithm: {algorithm}")
        try:
            # User selection for feature extraction methods
            if args.feature_methods:
                display_options(AUDIO_ALGORITHMS[algorithm]['features'], f"Available Feature Extraction Methods for {algorithm}")
                selected_features = get_user_selection(AUDIO_ALGORITHMS[algorithm]['features'], "Enter the numbers of the feature extraction methods you want to use (space-separated) or 'all': ")
            else:
                selected_features = AUDIO_ALGORITHMS[algorithm]['features']
                print(f"Using all available feature extraction methods for {algorithm}.")

            feature_subsets = audio_processor.run_analysis(algorithm, args.audio_path, selected_features, force_reprocess=args.force_reprocess)

            if feature_subsets.empty:
                print(f"No features extracted for {algorithm}. Skipping this algorithm.")
                continue

            print(f"Loading and merging data for {algorithm}...")
            merged_data = load_and_merge_data(LABELS_FILE_NAMES, feature_subsets)

            print(f"Splitting data for {algorithm}...")
            X = merged_data.drop(['Participant_ID'] + TARGET_COLUMNS, axis=1)
            y = merged_data[TARGET_COLUMNS]

            X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.2, random_state=42)

            print("Performing unsupervised feature selection...")
            selected_feature_names, X_selected = feature_selector.unsupervised_feature_selection(X, X.columns)

            # Use only selected feature names for all datasets
            X_train_selected = X_train[selected_feature_names]
            X_dev_selected = X_dev[selected_feature_names]
            X_test_selected = X_test[selected_feature_names]

            for target in TARGET_COLUMNS:
                print(f"Running ML models for target: {target}")
                y_train_target = y_train[target]
                y_dev_target = y_dev[target]
                y_test_target = y_test[target]

                # User selection for ML algorithms
                if args.ml_algorithms:
                    display_options(list(ML_ALGORITHMS.keys()), "Available ML Algorithms")
                    selected_ml_algorithms = get_user_selection(list(ML_ALGORITHMS.keys()), "Enter the numbers of the ML algorithms you want to use (space-separated) or 'all': ")
                else:
                    selected_ml_algorithms = list(ML_ALGORITHMS.keys())
                    print("Using all available ML algorithms.")

                ml_results = ml_processor.run_ml_pipeline(
                    X_train_selected, y_train_target, X_dev_selected, y_dev_target, X_test_selected, y_test_target, 
                    selected_ml_algorithms, algorithm=algorithm, sub_algorithm=target, target=target, use_cache=args.use_cache
                )

                if algorithm not in all_results[target]:
                    all_results[target][algorithm] = {}
                for feature in selected_feature_names:
                    all_results[target][algorithm][feature] = ml_results

                print(f"Features used for {algorithm}, {target}:", all_results[target][algorithm].keys())
                
            for target in TARGET_COLUMNS:
                print(f"\nGenerating 3D plot for {target}...")
                plot_path = os.path.join(args.ml_output_dir, f"best_accuracies_3d_plot_{target}.png")
                best_combo = MLPlot.plot_best_accuracies_3d(all_results[target], args.ml_output_dir, target)
                if best_combo:
                    print(f"Best combination for {target}: {best_combo}")
                    print(f"\nBest overall combination for {target}:")
                    print(f"Feature Extraction Algorithm: {best_combo['algo']}")
                    print(f"Feature Extraction Method: {best_combo['feature']}")
                    print(f"ML Algorithm: {best_combo['model']}")
                    print(f"Accuracy: {best_combo['accuracy']:.4f}")
                    print(f"3D plot saved to: {plot_path}")
                else:
                    print(f"No valid data to determine the best combination for {target}.")

        except Exception as e:
            print(f"Error processing algorithm {algorithm}: {str(e)}")
            continue

if __name__ == "__main__":
    
    # REQUIRED
    parser = argparse.ArgumentParser(description="Run audio processing and ML pipeline.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio files directory")
    
    # OPTIONAL
    parser.add_argument("--ml_output_dir", type=str, default="./output/plots", help="Output directory for ML results")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of audio files even if output exists")
    parser.add_argument("--no_cache", action="store_false", dest="use_cache", default=True,
                        help="Disable using cached ML models (caching is enabled by default)")
    parser.add_argument("--audio_algorithms", action="store_true", 
                        help="Enable selection of audio algorithms. If not specified, all algorithms will be used.")
    parser.add_argument("--feature_methods", action="store_true", 
                        help="Enable selection of feature extraction methods. If not specified, all methods will be used.")
    parser.add_argument("--ml_algorithms", action="store_true", 
                        help="Enable selection of ML algorithms. If not specified, all algorithms will be used.")
    parser.add_argument("--default", action="store_true", 
                        help="Use all default options without prompts. This overrides --audio_algorithms, --feature_methods, and --ml_algorithms.")
    args = parser.parse_args()

    if args.default:
        args.audio_algorithms = args.feature_methods = args.ml_algorithms = False

    main(args)