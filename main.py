import argparse
import os
import logging
import numpy as np
from data_loader import load_data, combine_features
from model_trainer import train_and_evaluate_models
from predictor import predict_on_test_data
from ml_plot import MLPlot
from config import TARGET_COLUMNS
from utils import setup_output_directories, save_json, calculate_weighted_accuracies

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train models and make predictions.")
    parser.add_argument("train_data_path", help="Path to the folder containing the training data files")
    parser.add_argument("prediction_data_path", help="Path to the folder containing the prediction data files")
    return parser.parse_args()

def load_and_preprocess_data(data_path, is_training=True):
    if is_training:
        labels = load_data(os.path.join(data_path, 'labels_complete.csv'))
        if labels is None:
            logging.error("Failed to load labels for training data. Exiting.")
            return None
    else:
        labels = None  # No labels for prediction data

    audio_features = load_data(os.path.join(data_path, 'audio_features.csv'))
    nlp_features = load_data(os.path.join(data_path, 'nlp_features.csv'))
    graph_features = load_data(os.path.join(data_path, 'graph_features.csv'))

    if audio_features is None or nlp_features is None or graph_features is None:
        logging.error("Failed to load one or more feature datasets. Exiting.")
        return None

    feature_sets = {
        'audio': audio_features.set_index('Participant_ID'),
        'nlp': nlp_features.set_index('Participant_ID'),
        'graph': graph_features.set_index('Participant_ID')
    }

    return labels, feature_sets

def main():
    args = parse_arguments()
    setup_output_directories()

    # Load and preprocess training data
    logging.info("Loading and preprocessing training data...")
    train_data = load_and_preprocess_data(args.train_data_path, is_training=True)
    if train_data is None:
        return

    labels, feature_sets = train_data

    # Train and evaluate models
    logging.info("Training and evaluating models...")
    all_results = train_and_evaluate_models(labels, feature_sets)

    # Calculate weighted accuracies
    weighted_accuracies = calculate_weighted_accuracies(all_results)
    f1_weighted_accuracies = calculate_weighted_accuracies(all_results, metric='test_f1')

    # Generate plots
    logging.info("Generating plots...")
    ml_plot = MLPlot()
    ml_plot.generate_plots(all_results, weighted_accuracies, f1_weighted_accuracies, './output/plots')

    # Save weighted accuracies
    save_json(weighted_accuracies, './output/weighted_accuracies.json')
    save_json(f1_weighted_accuracies, './output/f1_weighted_accuracies.json')

    # Load and preprocess prediction data
    logging.info("Loading and preprocessing prediction data...")
    prediction_data = load_and_preprocess_data(args.prediction_data_path, is_training=False)
    if prediction_data is None:
        return

    _, prediction_feature_sets = prediction_data

    # Make predictions on prediction data
    logging.info("Making predictions on prediction data...")
    predict_on_test_data(prediction_feature_sets, all_results)

    logging.info("Process completed successfully.")

if __name__ == "__main__":
    main()