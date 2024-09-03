from sklearn.ensemble import GradientBoostingClassifier
from .ml_model_base import MLModel

class GradientBoosting(MLModel):
    def __init__(self, params_file):
        super().__init__(GradientBoostingClassifier, params_file)