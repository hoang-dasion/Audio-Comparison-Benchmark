import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from plot.ml_plot import MLPlot
from const import TARGET_COLUMNS
from utils import (
    create_time_based_output_directory, check_audio_files, check_label_files,
    get_user_choices, process_algorithm
)

def process_algorithm_wrapper(args):
    return process_algorithm(*args)

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

    user_choices = get_user_choices(args)

    all_results = {target: {} for target in TARGET_COLUMNS}

    # Create a process pool
    num_processes = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for algorithm in user_choices['audio_algorithms']:
            future = executor.submit(
                process_algorithm_wrapper,
                (algorithm, args, user_choices['feature_methods'][algorithm], user_choices['ml_algorithms'])
            )
            futures.append(future)

        # Wait for all futures to complete
        for future in futures:
            result = future.result()
            if result:
                algorithm, algorithm_results = result
                for target in TARGET_COLUMNS:
                    all_results[target][algorithm] = algorithm_results[target]

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
