from sklearn.linear_model import LogisticRegression
from .ml_model_base import MLModel

class LogisticReg(MLModel):
    def __init__(self, params_file):
        super().__init__(LogisticRegression, params_file)