from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .ml_model_base import MLModel
from sklearn.svm import SVC

class SVM(MLModel):
    def __init__(self, params_file):
        super().__init__(SVC, params_file)

    def create_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**self.hyperparams))
        ])