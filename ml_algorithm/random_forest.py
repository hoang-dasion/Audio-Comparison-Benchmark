from sklearn.ensemble import RandomForestClassifier
from .ml_model_base import MLModel

class RandomForest(MLModel):
    def __init__(self, params_file):
        super().__init__(RandomForestClassifier, params_file)

    def create_pipeline(self):
        return super().create_pipeline()  # Ensure this returns a Pipeline object