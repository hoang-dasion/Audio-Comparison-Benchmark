import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from scipy.stats import loguniform
from tqdm import tqdm
import logging

class CustomLogisticRegression(LogisticRegression):
    def __init__(self, tol=1e-4, C=1.0, max_iter=100, random_state=None, solver='lbfgs', penalty='l2'):
        super().__init__(tol=tol, C=C, max_iter=max_iter, random_state=random_state, solver=solver, penalty=penalty)
        self.n_iter_ = 0
        self.loss_history_ = []
        self.classes_ = None  # Initialize classes_ attribute

    def fit(self, X, y):
        # Initialize SGD classifier with logistic loss
        sgd = SGDClassifier(loss='log_loss', penalty=self.penalty, alpha=1/(self.C * len(X)),
                            max_iter=1, learning_rate='optimal', random_state=self.random_state)

        # Set the unique classes and store them in the classes_ attribute
        self.classes_ = np.unique(y)
        
        for _ in range(self.max_iter):
            # Fit the model incrementally on the data
            sgd.partial_fit(X, y, classes=self.classes_)  
            
            # Store predictions to calculate the loss
            y_proba = sgd.predict_proba(X)
            
            # Calculate the current loss (log loss)
            current_loss = log_loss(y, y_proba)
            self.loss_history_.append(current_loss)
            
            prev_loss = self.loss_history_[-2] if len(self.loss_history_) > 1 else None
            
            self.n_iter_ += 1
            
            # Check for convergence
            if prev_loss is not None:
                relative_change = abs(prev_loss - current_loss) / abs(prev_loss)
                if relative_change < self.tol:
                    break
        
        # Store the model coefficients and intercept
        self.coef_ = sgd.coef_
        self.intercept_ = sgd.intercept_
        return self


def get_param_grid(algo_name):
    if algo_name == 'SVM':
        return {
            'C': loguniform(1e-3, 1e3),
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'] + list(loguniform(1e-4, 1e0).rvs(10)),
        }
    elif algo_name == 'Random Forest':
        return {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif algo_name == 'Gradient Boosting':
        return {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': loguniform(0.001, 0.1),
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif algo_name == 'K-Nearest Neighbors':
        return {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    elif algo_name == 'Logistic Regression':
        return {
            'C': loguniform(1e-4, 1e4),
            'tol': loguniform(1e-6, 1e-2),
            'max_iter': [1000, 2000, 5000, 10000],
            'solver': ['saga'],
            'penalty': ['l1', 'l2']
        }
    elif algo_name == 'Naive Bayes':
        return {
            'var_smoothing': loguniform(1e-10, 1e-8),
        }
    elif algo_name == 'Decision Tree':
        return {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    elif algo_name == 'Multi-layer Perceptron':
        return {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['tanh', 'relu'],
            'alpha': loguniform(1e-5, 1e-2),
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [1000, 2000, 5000],
            'tol': loguniform(1e-6, 1e-3),
            'early_stopping': [True],
            'n_iter_no_change': [10, 20, 50]
        }
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def get_model(algo_name):
    if algo_name == 'SVM':
        return SVC(random_state=42)
    elif algo_name == 'Random Forest':
        return RandomForestClassifier(random_state=42)
    elif algo_name == 'Gradient Boosting':
        return GradientBoostingClassifier(random_state=42)
    elif algo_name == 'K-Nearest Neighbors':
        return KNeighborsClassifier()
    elif algo_name == 'Logistic Regression':
        return CustomLogisticRegression(random_state=42)
    elif algo_name == 'Naive Bayes':
        return GaussianNB()
    elif algo_name == 'Decision Tree':
        return DecisionTreeClassifier(random_state=42)
    elif algo_name == 'Multi-layer Perceptron':
        return MLPClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def optimize_params(algo_name, X, y):
    param_grid = get_param_grid(algo_name)
    model = get_model(algo_name)
    
    n_iter = min(100, len(param_grid))  # Set n_iter to be the smaller of 100 or the total parameter combinations
    n_jobs = -1
    
    # RandomizedSearchCV will handle the fitting internally without a need for tqdm
    search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=5, n_jobs=n_jobs, random_state=42, verbose=0)
    
    search.fit(X, y)  # This will now run without the tqdm progress bar
    
    return search.best_params_