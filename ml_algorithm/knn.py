from sklearn.neighbors import KNeighborsClassifier
from .ml_model_base import MLModel

class KNN(MLModel):
    def __init__(self, params_file):
        super().__init__(KNeighborsClassifier, params_file)