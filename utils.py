import os
import json
import numpy as np
from const import ALL_MODELS, FEATURES_DIC

from ml_algorithm.svm import SVM
from ml_algorithm.random_forest import RandomForest
from ml_algorithm.gradient_boosting import GradientBoosting
from ml_algorithm.knn import KNN
from ml_algorithm.logistic_regression import LogisticReg
from ml_algorithm.naive_bayes import NaiveBayes
from ml_algorithm.decision_tree import DecisionTree
from ml_algorithm.mlp import MLP

def get_models(output_dir):
    return {
        'SVM': SVM(f"{output_dir}/params/SVM/svm_params.json"),
        'Random Forest': RandomForest(f"{output_dir}/params/Random Forest/rf_params.json"),
        'Gradient Boosting': GradientBoosting(f"{output_dir}/params/Gradient Boosting/gb_params.json"),
        'K-Nearest Neighbors': KNN(f"{output_dir}/params/K-Nearest Neighbors/knn_params.json"),
        'Logistic Regression': LogisticReg(f"{output_dir}/params/Logistic Regression/lr_params.json"),
        'Naive Bayes': NaiveBayes(f"{output_dir}/params/Naive Bayes/nb_params.json"),
        'Decision Tree': DecisionTree(f"{output_dir}/params/Decision Tree/dt_params.json"),
        'Multi-layer Perceptron': MLP(f"{output_dir}/params/Multi-layer Perceptron/mlp_params.json")
    }

def save_config(algorithm, runtimes, durations, feature_names, params, metrics, output_dir, audio_path):
    config_info = f"""{algorithm.replace("_", " ").title()} Analysis
    metrics: {metrics}
    -------------------------------
    runtime: {runtimes}
    durations: {durations}
    feature_names: {feature_names}
    params: {params}
    -------------------------------
    output_dir: {output_dir}
    audio_path: {audio_path}
    """

    config_dir = os.path.join(output_dir, algorithm, 'config')
    os.makedirs(config_dir, exist_ok=True)
    config_file_path = os.path.join(config_dir, 'config.txt')
    with open(config_file_path, 'w') as f:
        f.write(config_info)

def save_params(params, params_file, output_dir, algorithm):
    params_file = params_file if params_file.endswith('.json') else f"{params_file}.json"
    params_dir = os.path.join(output_dir, algorithm, 'params')
    os.makedirs(params_dir, exist_ok=True)

    params_file_path = os.path.join(params_dir, params_file)
    with open(params_file_path, 'w') as f:
        json.dump(params, f, indent=4)
