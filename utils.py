import json
from ml_algorithm.svm import SVM
from ml_algorithm.random_forest import RandomForest
from ml_algorithm.gradient_boosting import GradientBoosting
from ml_algorithm.knn import KNN
from ml_algorithm.logistic_regression import LogisticReg
from ml_algorithm.naive_bayes import NaiveBayes
from ml_algorithm.decision_tree import DecisionTree
from ml_algorithm.mlp import MLP

def get_models(output_dir):
    return {
        'SVM': SVM(f"{output_dir}/params/SVM/svm_params.json"),
        'Random Forest': RandomForest(f"{output_dir}/params/Random Forest/rf_params.json"),
        'Gradient Boosting': GradientBoosting(f"{output_dir}/params/Gradient Boosting/gb_params.json"),
        'K-Nearest Neighbors': KNN(f"{output_dir}/params/K-Nearest Neighbors/knn_params.json"),
        'Logistic Regression': LogisticReg(f"{output_dir}/params/Logistic Regression/lr_params.json"),
        'Naive Bayes': NaiveBayes(f"{output_dir}/params/Naive Bayes/nb_params.json"),
        'Decision Tree': DecisionTree(f"{output_dir}/params/Decision Tree/dt_params.json"),
        'Multi-layer Perceptron': MLP(f"{output_dir}/params/Multi-layer Perceptron/mlp_params.json")
    }