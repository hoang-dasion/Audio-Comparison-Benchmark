from sklearn.ensemble import GradientBoostingClassifier
from .ml_model_base import MLModel

class GradientBoosting(MLModel):
    def __init__(self, params_file):
        super().__init__(GradientBoostingClassifier, params_file)

    def create_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingClassifier(**self.hyperparams))
        ])