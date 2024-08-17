# ml_algorithm/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from .ml_model_base import MLModel

class RandomForest(MLModel):
    def __init__(self, params_file):
        super().__init__(RandomForestClassifier, params_file)