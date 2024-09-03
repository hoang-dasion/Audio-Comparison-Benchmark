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
        dir_name = ''.join(['_'+c.lower() if c.isupper() else c for c in self.model_class.__name__]).lstrip('_')
        params_dir = os.path.join('ml_algorithm', 'params', dir_name)
        os.makedirs(params_dir, exist_ok=True)
        params_path = os.path.join(params_dir, self.params_file)
        if not os.path.exists(params_path):
            print(f"File {params_path} not found. Creating with default parameters.")
            default_params = {}
            with open(params_path, 'w') as f:
                json.dump(default_params, f, indent=4)

        with open(params_path, 'r') as f:
            return json.load(f)

    def create_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model_class(**self.hyperparams))
        ])