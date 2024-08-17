from sklearn.tree import DecisionTreeClassifier
from .ml_model_base import MLModel

class DecisionTree(MLModel):
    def __init__(self, params_file):
        super().__init__(DecisionTreeClassifier, params_file)