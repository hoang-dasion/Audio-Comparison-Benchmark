import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from processor.audio_processor import AudioProcessor
from processor.feature_selection_processor import FeatureSelectionProcessor
from processor.ml_processor import MLProcessor
from plot.ml_plot import MLPlot
from const import FEATURES_DIC, TARGET_COLUMNS, ALL_MODELS, LABELS_FILE_NAMES, LABELS_INPUT_DIR, LABELS_OUTPUT_DIR
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_algorithm(algorithm, args, audio_processor, feature_selector, ml_processor):
    print(f"\nProcessing algorithm: {algorithm}")
    feature_subsets = audio_processor.run_analysis(algorithm, args.audio_path, force_reprocess=args.force_reprocess)

    if feature_subsets.empty:
        print(f"No features extracted for {algorithm}. Skipping this algorithm.")
        return None

    print(f"Loading and merging data for {algorithm}...")
    merged_data = load_and_merge_data(LABELS_FILE_NAMES, feature_subsets)

    if merged_data.shape[0] == 0:
        print(f"Error: No data available after merging for {algorithm}. Skipping this algorithm.")
        return None

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
            ALL_MODELS, algorithm=algorithm, target=target, use_cache=args.use_cache
        )

        algorithm_results[target] = {feature: ml_results for feature in selected_feature_names}

    return algorithm, algorithm_results

def load_and_merge_data(label_files, feature_subsets):
    all_labels = pd.concat([pd.read_csv(os.path.join(LABELS_OUTPUT_DIR, f"complete_{file}.csv")) for file in label_files])
    all_labels['Participant_ID'] = all_labels['Participant_ID'].astype(str)
    feature_subsets['Participant_ID'] = feature_subsets['Participant_ID'].astype(str)
    
    merged_data = pd.merge(feature_subsets, all_labels, on='Participant_ID', how='inner')
        
    if merged_data.shape[0] == 0:
        print("Warning: No data after merging. Checking for mismatched Participant_IDs...")
        label_ids = set(all_labels['Participant_ID'])
        feature_ids = set(feature_subsets['Participant_ID'])
    
    return merged_data

def main(args):
    print("Initializing processors...")
    audio_processor = AudioProcessor()
    feature_selector = FeatureSelectionProcessor()
    ml_processor = MLProcessor(args.ml_output_dir)

    all_results = {target: {} for target in TARGET_COLUMNS}

    with ThreadPoolExecutor(max_workers=min(len(args.algorithms), os.cpu_count())) as executor:
        future_to_algorithm = {executor.submit(process_algorithm, algorithm, args, audio_processor, feature_selector, ml_processor): algorithm for algorithm in args.algorithms}
        
        for future in as_completed(future_to_algorithm):
            algorithm = future_to_algorithm[future]
            try:
                result = future.result()
                if result:
                    algorithm, algorithm_results = result
                    for target in TARGET_COLUMNS:
                        all_results[target][algorithm] = algorithm_results[target]
            except Exception as exc:
                print(f'{algorithm} generated an exception: {exc}')

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
    parser.add_argument("--algorithms", nargs='+', default=list(FEATURES_DIC.keys()),
                        choices=list(FEATURES_DIC.keys()),
                        help="Audio processing algorithms (default: all)")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio files directory")
    parser.add_argument("--ml_output_dir", type=str, default="./ml_output", help="Output directory for ML results")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of audio files even if output exists")
    parser.add_argument("--no_cache", action="store_false", dest="use_cache", default=True,
                        help="Disable using cached ML models (caching is enabled by default)")
    args = parser.parse_args()
    main(args)