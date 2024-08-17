import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from const import ALL_MODELS
from utils import get_models
from plot.ml_plot import MLPlot

from tqdm import tqdm

class MLProcessor:
    def __init__(self, output_dir='ML'):
        self.output_dir = output_dir
        self.models = get_models(output_dir)

    def train_and_predict(self, X, y, model_name, cv=3, algorithm='', sub_algorithm=''):
        base_dir = f"{self.output_dir}/plots/{algorithm}/{sub_algorithm}"
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = self.models[model_name]
        pipeline = model.create_pipeline()
        
        pipeline.fit(X_train, y_train)
        
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
                
        cv_scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), scoring='accuracy')
        
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_test)
        classes = np.unique(y)
        
        MLPlot.plot_confusion_matrix(cm_train, classes, f"{base_dir}/confusion_matrix/{model_name}_train_cm.png")
        MLPlot.plot_confusion_matrix(cm_test, classes, f"{base_dir}/confusion_matrix/{model_name}_test_cm.png")
        
        # print(classification_report(y_test, y_pred_test, zero_division=0))
        
        return pipeline, train_accuracy, test_accuracy

    def run_ml_pipeline(self, X, y, selected_models, algorithm='', sub_algorithm=''):
        train_accuracies = []
        test_accuracies = []
        model_names = []
        
        for model_name in tqdm(selected_models, desc="Training ML models"):
            
            model, train_accuracy, test_accuracy = self.train_and_predict(X, y, model_name, algorithm=algorithm, sub_algorithm=sub_algorithm)
            
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            model_names.append(model_name)
        
        plot_output_dir = f"{self.output_dir}/plots/{algorithm}/{sub_algorithm}/accuracy"

        MLPlot.plot_grouped_accuracy_comparison(train_accuracies, test_accuracies, model_names, f"{plot_output_dir}/grouped_train_test_comparison.png")