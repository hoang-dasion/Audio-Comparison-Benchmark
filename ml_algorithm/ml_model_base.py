import os
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model_class(**self.hyperparams))
        ])