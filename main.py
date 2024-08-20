import argparse
from processor.audio_processor import AudioProcessor
from processor.ml_processor import MLProcessor
from const import FEATURES_DIC, ALL_MODELS
from plot.ml_plot import MLPlot
import json

from tqdm import tqdm
import numpy as np

def process_audio_and_run_ml(args):

    audio_processor = AudioProcessor()
    ml_processor = MLProcessor(output_dir=args.ml_output_dir)

    all_results = {}
    for algorithm in args.algorithms:
        feature_arrays, labels = audio_processor.run_analysis(
            algorithm=algorithm,
            runtimes=args.runtimes,
            durations=args.durations,
            feature_names=args.features.get(algorithm, None),
            metrics=args.metrics,
            output_dir=args.output_dir,
            params_file=args.params_file,
            audio_path=args.audio_path
        )

        # Select models based on command line arguments
        if 'all' in args.ml:
            selected_models = ALL_MODELS
        else:
            selected_models = [model for model in ALL_MODELS if model.lower().split()[0] in args.ml]

        all_results[algorithm] = {}
        for feature_name, feature_array in feature_arrays.items():
            X = feature_array.reshape(feature_array.shape[0], -1)
            y = labels

            print(f"\nRunning ML pipeline for {algorithm} - {feature_name}")
            ml_results = ml_processor.run_ml_pipeline(X, y, selected_models, algorithm=algorithm, sub_algorithm=feature_name)
            
            all_results[algorithm][feature_name] = ml_results

    # After all processing is done, create the 3D plot
    best_combo = MLPlot.plot_best_accuracies_3d(all_results, args.ml_output_dir, FEATURES_DIC)
    return best_combo

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run audio processing and ML pipeline.")
    
    # Arguments with required params
    parser.add_argument("--algorithms", nargs='+', default=list(FEATURES_DIC.keys()),
                        choices=list(FEATURES_DIC.keys()),
                        help="Audio processing algorithms (default: all)")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file or directory")
    
    # Arguments with default params
    parser.add_argument("--runtimes", type=int, default=1, help="Number of runtimes")
    parser.add_argument("--durations", type=str, default="full", help="Comma-separated list of durations")
    parser.add_argument("--features", type=json.loads, default={}, 
                        help="JSON string of algorithms and their features to extract")
    parser.add_argument("--metrics", nargs='+', default=None, help="List of metrics to compute")
    parser.add_argument("--output_dir", type=str, default="./audio_output", help="Output directory for audio processing")
    parser.add_argument("--ml_output_dir", type=str, default="./ml_output", help="Output directory for ML results")
    parser.add_argument("--params_file", type=str, default="params", help="Parameters file")
    parser.add_argument("-ml", nargs='+', default=['all'], 
                        choices=['all'] + [model.lower().split()[0] for model in ALL_MODELS],
                        help="Specify which ML algorithms to run. Default is 'all'.")

    args = parser.parse_args()
    process_audio_and_run_ml(args)