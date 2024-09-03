# ml_processor.py

import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from const import ML_ALGORITHMS
from plot.ml_plot import MLPlot
import importlib

class MLProcessor:
    def __init__(self, cache_dir, plots_dir):
        self.cache_dir = cache_dir
        self.plots_dir = plots_dir
        self.models = self._load_models()

    def _load_models(self):
        models = {}
        for model_name, model_info in ML_ALGORITHMS.items():
            module = importlib.import_module(f"ml_algorithm.{model_info['file'][:-3]}")
            model_class = getattr(module, model_info['class'])
            params_file = f"{model_name.lower().replace(' ', '_')}_params.json"
            models[model_name] = model_class(params_file)
        return models

    def save_model(self, model, model_name, algorithm, sub_algorithm, target):
        model_filename = f"{algorithm}_{sub_algorithm}_{target}_{model_name.lower().replace(' ', '_')}.pkl"
        model_path = os.path.join(self.cache_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, model_name, algorithm, sub_algorithm, target):
        model_filename = f"{algorithm}_{sub_algorithm}_{target}_{model_name.lower().replace(' ', '_')}.pkl"
        model_path = os.path.join(self.cache_dir, model_filename)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None

    def train_and_predict(self, X_train, y_train, X_dev, y_dev, X_test, y_test, model_name, algorithm='', sub_algorithm='', target='', use_cache=False):
        algorithm_plots_dir = os.path.join(self.plots_dir, algorithm, sub_algorithm)
        os.makedirs(algorithm_plots_dir, exist_ok=True)

        if use_cache:
            cached_model = self.load_model(model_name, algorithm, sub_algorithm, target)
            pipeline = cached_model if cached_model is not None else None
        else:
            pipeline = None

        if pipeline is None:
            model = self.models[model_name]
            pipeline = model.create_pipeline()
            pipeline.fit(X_train, y_train)
            self.save_model(pipeline, model_name, algorithm, sub_algorithm, target)

        y_pred_train = pipeline.predict(X_train)
        y_pred_dev = pipeline.predict(X_dev)
        y_pred_test = pipeline.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        dev_accuracy = accuracy_score(y_dev, y_pred_dev)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_dev = confusion_matrix(y_dev, y_pred_dev)
        cm_test = confusion_matrix(y_test, y_pred_test)
        classes = np.unique(np.concatenate([y_train, y_dev, y_test]))

        confusion_matrix_dir = os.path.join(algorithm_plots_dir, "confusion_matrix")
        os.makedirs(confusion_matrix_dir, exist_ok=True)
        MLPlot.plot_confusion_matrix(cm_train, classes, os.path.join(confusion_matrix_dir, f"{model_name}_train_cm.png"))
        MLPlot.plot_confusion_matrix(cm_dev, classes, os.path.join(confusion_matrix_dir, f"{model_name}_dev_cm.png"))
        MLPlot.plot_confusion_matrix(cm_test, classes, os.path.join(confusion_matrix_dir, f"{model_name}_test_cm.png"))

        return pipeline, train_accuracy, dev_accuracy, test_accuracy

    def run_ml_pipeline(self, X_train, y_train, X_dev, y_dev, X_test, y_test, selected_models, algorithm='', sub_algorithm='', target='', use_cache=False):
        results = {}
        
        for model_name in selected_models:
            try:
                model, train_accuracy, dev_accuracy, test_accuracy = self.train_and_predict(
                    X_train, y_train, X_dev, y_dev, X_test, y_test, model_name, 
                    algorithm=algorithm, sub_algorithm=sub_algorithm, target=target, use_cache=use_cache
                )

                results[model_name] = {
                    'train_accuracy': train_accuracy,
                    'dev_accuracy': dev_accuracy,
                    'test_accuracy': test_accuracy,
                    'model': model
                }
            except Exception as e:
                print(f"Error occurred while training {model_name}: {str(e)}")
                results[model_name] = {
                    'train_accuracy': None,
                    'dev_accuracy': None,
                    'test_accuracy': None,
                    'model': None,
                    'error': str(e)
                }

        accuracy_plot_dir = os.path.join(self.plots_dir, algorithm, sub_algorithm, "accuracy")
        os.makedirs(accuracy_plot_dir, exist_ok=True)
        MLPlot.plot_grouped_accuracy_comparison(
            [results[model]['train_accuracy'] for model in selected_models if results[model]['train_accuracy'] is not None],
            [results[model]['dev_accuracy'] for model in selected_models if results[model]['dev_accuracy'] is not None],
            [results[model]['test_accuracy'] for model in selected_models if results[model]['test_accuracy'] is not None],
            [model for model in selected_models if results[model]['train_accuracy'] is not None],
            os.path.join(accuracy_plot_dir, "grouped_train_dev_test_comparison.png")
        )

        return results