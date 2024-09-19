import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_loader import combine_features, generate_feature_combinations
from config import ML_ALGORITHMS, TARGET_COLUMNS
import logging

def load_selected_features(target, combo):
    features_file = f'./output/selected_features/{target}/{"-".join(sorted(combo))}_features.json'
    with open(features_file, 'r') as f:
        return json.load(f)

def load_scaler(target, combo):
    scaler_file = f'./output/saved_scalers/{target}/{"-".join(sorted(combo))}_scaler.joblib'
    return joblib.load(scaler_file)

def load_model(target, combo, algo_name):
    model_file = f'./output/saved_models/{target}/{algo_name}_{"-".join(sorted(combo))}.joblib'
    return joblib.load(model_file)

def predict_single_model(model, X_scaled):
    return model.predict(X_scaled)

def predict_on_test_data(test_feature_sets, all_results):
    all_predictions = {target: {} for target in TARGET_COLUMNS}
    
    for target in TARGET_COLUMNS:
        feature_combinations = generate_feature_combinations(test_feature_sets)
        for combo in tqdm(feature_combinations, desc=f"Processing combinations for {target}"):
            selected_features = load_selected_features(target, combo)
            
            combined_features = combine_features([test_feature_sets[feat] for feat in combo])
            X = combined_features.reindex(columns=selected_features, fill_value=0)
            
            scaler = load_scaler(target, combo)
            X_scaled = scaler.transform(X)
            
            combo_predictions = {}
            for algo_name in ML_ALGORITHMS:
                model = load_model(target, combo, algo_name)
                
                try:
                    predictions = predict_single_model(model, X_scaled)
                    combo_predictions[algo_name] = {
                        participant_id: int(prediction)
                        for participant_id, prediction in zip(X.index, predictions)
                    }
                except Exception as e:
                    logging.error(f"Error making predictions for {target}, {'-'.join(combo)}, {algo_name}: {str(e)}")
            
            all_predictions[target]['-'.join(sorted(combo))] = combo_predictions

    save_predictions(all_predictions)

def save_predictions(all_predictions):
    with open('./output/test_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=4)