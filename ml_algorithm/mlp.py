from sklearn.neural_network import MLPClassifier
from .ml_model_base import MLModel

class MLP(MLModel):
    def __init__(self, params_file):
        super().__init__(MLPClassifier, params_file)    