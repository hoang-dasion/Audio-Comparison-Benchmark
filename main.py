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

    algorithms_to_process = args.algorithms if args.algorithms else list(AUDIO_ALGORITHMS.keys())

    for algorithm in algorithms_to_process:
        print(f"\nProcessing algorithm: {algorithm}")
        try:
            feature_subsets = audio_processor.run_analysis(algorithm, args.audio_path, force_reprocess=args.force_reprocess)

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

                ml_results = ml_processor.run_ml_pipeline(
                    X_train_selected, y_train_target, X_dev_selected, y_dev_target, X_test_selected, y_test_target, 
                    list(ML_ALGORITHMS.keys()), algorithm=algorithm, sub_algorithm=target, target=target, use_cache=args.use_cache
                )

                if algorithm not in all_results[target]:
                    all_results[target][algorithm] = {}
                for feature in selected_feature_names:
                    all_results[target][algorithm][feature] = ml_results

                print(f"Features used for {algorithm}, {target}:", all_results[target][algorithm].keys())

        except Exception as e:
            print(f"Error processing algorithm {algorithm}: {str(e)}")
            continue

    for target in TARGET_COLUMNS:
        print(f"\nGenerating 3D plot for {target}...")
        best_combo = MLPlot.plot_best_accuracies_3d(all_results[target], args.ml_output_dir, target)
        if best_combo:
            print(f"Best combination for {target}: {best_combo}")
            print(f"\nBest overall combination for {target}:")
            print(f"Feature Extraction Algorithm: {best_combo['algo']}")
            print(f"Feature Extraction Method: {best_combo['feature']}")
            print(f"ML Algorithm: {best_combo['model']}")
            print(f"Accuracy: {best_combo['accuracy']:.4f}")
        else:
            print(f"No valid data to determine the best combination for {target}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio processing and ML pipeline.")
    parser.add_argument("--algorithms", nargs='+', default=None,
                        choices=list(AUDIO_ALGORITHMS.keys()),
                        help="Audio processing algorithms (default: all)")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio files directory")
    parser.add_argument("--ml_output_dir", type=str, default="./output", help="Output directory for ML results")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of audio files even if output exists")
    parser.add_argument("--no_cache", action="store_false", dest="use_cache", default=True,
                        help="Disable using cached ML models (caching is enabled by default)")
    args = parser.parse_args()
    main(args)