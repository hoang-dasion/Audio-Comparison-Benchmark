import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from data_loader import combine_features, generate_feature_combinations
from feature_selection import perform_feature_selection
from ml_algorithms import MLAlgorithms
from config import ML_ALGORITHMS, TARGET_COLUMNS
from utils import timing_decorator, save_json
from optimize_params import optimize_params
import logging

def load_ml_params(algo_name):
    param_file = f"./params/{algo_name.lower().replace(' ', '_')}_params.json"
    try:
        with open(param_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"No parameter file found for {algo_name}. Using default parameters.")
        return {}

@timing_decorator
def train_and_evaluate_model(algo_name, X_train, y_train, X_test, y_test, params):
    ml_algorithms = MLAlgorithms()
    model = ml_algorithms.train_model(algo_name, X_train, y_train, params)
    train_accuracy, test_accuracy, train_f1, test_f1 = ml_algorithms.evaluate_model(model, X_train, y_train, X_test, y_test)
    test_pred = model.predict(X_test)
    return train_accuracy, test_accuracy, train_f1, test_f1, test_pred, model

def train_and_evaluate_models(labels, feature_sets, optimize=False):
    all_results = {target: {} for target in TARGET_COLUMNS}

    for target in TARGET_COLUMNS:
        feature_combinations = generate_feature_combinations(feature_sets)
        for combo in tqdm(feature_combinations, desc=f"Processing combinations for {target}"):
            combined_features = combine_features([feature_sets[feat] for feat in combo])
            data = pd.merge(combined_features, labels[['Participant_ID', target]], left_index=True, right_on='Participant_ID')
            
            X = data.drop(['Participant_ID', target], axis=1)
            y = data[target]
            
            if X.empty:
                logging.warning(f"X is empty for combination {combo}")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            selected_features, X_train_selected, feature_selection_info = perform_feature_selection(X_train, y_train)
            X_test_selected = X_test[selected_features]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            combo_results = {}
            for algo_name in ML_ALGORITHMS:
                if optimize:
                    best_params = optimize_params(algo_name, X_train_scaled, y_train)
                else:
                    # Use default or pre-optimized parameters
                    best_params = load_ml_params(algo_name)
                
                result, execution_time = train_and_evaluate_model(
                    algo_name, X_train_scaled, y_train, X_test_scaled, y_test, best_params
                )
                train_accuracy, test_accuracy, train_f1, test_f1, test_pred, model = result
                
                combo_results[algo_name] = {
                    'train_accuracy': float(train_accuracy),
                    'test_accuracy': float(test_accuracy),
                    'train_f1': float(train_f1),
                    'test_f1': float(test_f1),
                    'test_pred': test_pred.tolist(),
                    'execution_time': execution_time,
                    'best_params': best_params
                }

                save_model(model, target, algo_name, combo)
                save_selected_features(selected_features, target, combo)
                save_scaler(scaler, target, combo)
                save_feature_selection_info(feature_selection_info, target, combo)

            all_results[target]['-'.join(sorted(combo))] = combo_results

    save_results(all_results)
    return all_results

def save_model(model, target, algo_name, combo):
    model_dir = f'./output/saved_models/{target}'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f'{model_dir}/{algo_name}_{"-".join(sorted(combo))}.joblib')

def save_selected_features(selected_features, target, combo):
    features_dir = f'./output/selected_features/{target}'
    os.makedirs(features_dir, exist_ok=True)
    with open(f'{features_dir}/{"-".join(sorted(combo))}_features.json', 'w') as f:
        json.dump(selected_features, f)

def save_scaler(scaler, target, combo):
    scaler_dir = f'./output/saved_scalers/{target}'
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, f'{scaler_dir}/{"-".join(sorted(combo))}_scaler.joblib')

def save_feature_selection_info(feature_selection_info, target, combo):
    info_dir = f'./output/feature_selection_info/{target}'
    os.makedirs(info_dir, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj

    serializable_info = convert_to_serializable(feature_selection_info)
    
    with open(f'{info_dir}/{"-".join(sorted(combo))}_info.json', 'w') as f:
        json.dump(serializable_info, f, indent=4)

def save_results(all_results):
    save_json(all_results, './output/ml_output_results.json')