from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

class MLAlgorithms:
    def __init__(self):
        self.algorithms = {
            'SVM': SVC,
            'Random Forest': RandomForestClassifier,
            'Gradient Boosting': GradientBoostingClassifier,
            'K-Nearest Neighbors': KNeighborsClassifier,
            'Logistic Regression': LogisticRegression,
            'Naive Bayes': GaussianNB,
            'Decision Tree': DecisionTreeClassifier,
            'Multi-layer Perceptron': MLPClassifier
        }

    def train_model(self, algo_name, X_train, y_train, params=None):
        if params is None:
            params = {}
        model = self.algorithms[algo_name](**params)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        return train_accuracy, test_accuracy, train_f1, test_f1