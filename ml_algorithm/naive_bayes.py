from sklearn.naive_bayes import GaussianNB
from .ml_model_base import MLModel

class NaiveBayes(MLModel):
    def __init__(self, params_file):
        super().__init__(GaussianNB, params_file)