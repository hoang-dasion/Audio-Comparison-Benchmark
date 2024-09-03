import os
import argparse
from processor.audio_processor import AudioProcessor
from processor.feature_selection_processor import FeatureSelectionProcessor
from processor.ml_processor import MLProcessor
from plot.ml_plot import MLPlot
from const import AUDIO_ALGORITHMS, ML_ALGORITHMS, TARGET_COLUMNS
from utils import (
    create_time_based_output_directory, check_audio_files, check_label_files,
    display_options, get_user_selection, process_algorithm, get_user_choices
)

def main(args):
    print("Creating time-based output directory...")
    time_output_dir = create_time_based_output_directory()
    args.cached_models_dir = os.path.join(time_output_dir, "cached_models")
    args.plots_dir = os.path.join(time_output_dir, "plots")

    print("Checking audio files...")
    wav_files = check_audio_files(args.audio_path)
    print(f"Found {len(wav_files)} .wav files.")

    print("Checking label files...")
    check_label_files()

    print("Initializing processors...")
    audio_processor = AudioProcessor()
    feature_selector = FeatureSelectionProcessor()
    ml_processor = MLProcessor(args.cached_models_dir, args.plots_dir)

    all_results = {target: {} for target in TARGET_COLUMNS}

    user_choices = get_user_choices(args)

    for algorithm in user_choices['audio_algorithms']:
        try:
            algorithm_results = process_algorithm(
                algorithm, args, audio_processor, feature_selector, ml_processor,
                user_choices['feature_methods'][algorithm], user_choices['ml_algorithms']
            )

            if algorithm_results:
                for target in TARGET_COLUMNS:
                    all_results[target][algorithm] = algorithm_results[target]

        except Exception as e:
            print(f"Error processing algorithm {algorithm}: {str(e)}")
            continue

    for target in TARGET_COLUMNS:
        print(f"\nGenerating 3D plot for {target}...")
        plot_path = os.path.join(args.plots_dir, f"best_accuracies_3d_plot_{target}.png")
        best_combo = MLPlot.plot_best_accuracies_3d(all_results[target], args.plots_dir, target)
        if best_combo:
            print(f"Best combination for {target}: {best_combo}")
            print(f"3D plot saved to: {plot_path}")
        else:
            print(f"No valid data to determine the best combination for {target}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio processing and ML pipeline.")
    
    # REQUIRED
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio files directory")
    
    # OPTIONAL
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of audio files even if output exists")
    parser.add_argument("--no_cache", action="store_false", dest="use_cache", default=True,
                        help="Disable using cached ML models (caching is enabled by default)")
    parser.add_argument("--custom", action="store_true", 
                        help="Enable custom selection of algorithms and methods. If not specified, all options will be used.")
    args = parser.parse_args()

    main(args)