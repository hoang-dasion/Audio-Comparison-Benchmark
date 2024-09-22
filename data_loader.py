import pandas as pd
import logging
from itertools import combinations

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        return None

def combine_features(feature_sets):
    return pd.concat(feature_sets, axis=1)

def generate_feature_combinations(feature_sets):
    all_combinations = []
    for r in range(1, len(feature_sets) + 1):
        all_combinations.extend(combinations(feature_sets.keys(), r))
    return all_combinations