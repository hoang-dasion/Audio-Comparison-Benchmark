# ml_algorithm/ml_model_base.py

import os
import json
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class MLModel:
    def __init__(self, model_class, params_file):
        self.model_class = model_class
        self.params_file = params_file
        self.hyperparams = self.load_params()

    def load_params(self):
        os.makedirs(os.path.dirname(self.params_file), exist_ok=True)
        if not os.path.exists(self.params_file):
            print(f"File {self.params_file} not found. Creating with default parameters.")
            default_params = {}
            with open(self.params_file, 'w') as f:
                json.dump(default_params, f, indent=4)
        
        with open(self.params_file, 'r') as f:
            return json.load(f)

    def create_pipeline(self):
        if self.model_class in [RandomForestClassifier, GradientBoostingClassifier]:
            return Pipeline([
                ('model', self.model_class(**self.hyperparams))
            ])
        else:
            return Pipeline([
                ('imputer', IterativeImputer(random_state=42)),
                ('model', self.model_class(**self.hyperparams))
            ])