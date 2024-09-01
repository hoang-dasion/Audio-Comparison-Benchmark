import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import get_models
from plot.ml_plot import MLPlot
from concurrent.futures import ThreadPoolExecutor, as_completed

class MLProcessor:
    def __init__(self, output_dir='ML'):
        self.output_dir = output_dir
        self.models = get_models(output_dir)
        self.cache_dir = os.path.join(output_dir, 'cached_models')
        os.makedirs(self.cache_dir, exist_ok=True)

    def save_model(self, model, model_name, algorithm, sub_algorithm, target):
        model_path = os.path.join(self.cache_dir, f"{algorithm}_{sub_algorithm}_{target}_{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, model_name, algorithm, sub_algorithm, target):
        model_path = os.path.join(self.cache_dir, f"{algorithm}_{sub_algorithm}_{target}_{model_name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None

    def train_and_predict(self, X_train, y_train, X_dev, y_dev, X_test, y_test, model_name, algorithm='', sub_algorithm='', target='', use_cache=False):
        base_dir = f"{self.output_dir}/plots/{algorithm}/{sub_algorithm}"

        if use_cache:
            cached_model = self.load_model(model_name, algorithm, sub_algorithm, target)
            if cached_model is not None:
                print(f"Using cached model for {model_name}")
                pipeline = cached_model
            else:
                print(f"No cached model found for {model_name}. Training new model.")
                pipeline = None
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

        MLPlot.plot_confusion_matrix(cm_train, classes, f"{base_dir}/confusion_matrix/{model_name}_train_cm.png")
        MLPlot.plot_confusion_matrix(cm_dev, classes, f"{base_dir}/confusion_matrix/{model_name}_dev_cm.png")
        MLPlot.plot_confusion_matrix(cm_test, classes, f"{base_dir}/confusion_matrix/{model_name}_test_cm.png")

        return pipeline, train_accuracy, dev_accuracy, test_accuracy

    def run_ml_pipeline(self, X_train, y_train, X_dev, y_dev, X_test, y_test, selected_models, algorithm='', sub_algorithm='', target='', use_cache=False):
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(selected_models), os.cpu_count())) as executor:
            future_to_model = {executor.submit(self.train_and_predict, X_train, y_train, X_dev, y_dev, X_test, y_test, model_name, algorithm, sub_algorithm, target, use_cache): model_name for model_name in selected_models}
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    model, train_accuracy, dev_accuracy, test_accuracy = future.result()
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

        plot_output_dir = f"{self.output_dir}/plots/{algorithm}/{sub_algorithm}/accuracy"
        MLPlot.plot_grouped_accuracy_comparison(
            [results[model]['train_accuracy'] for model in selected_models if results[model]['train_accuracy'] is not None],
            [results[model]['dev_accuracy'] for model in selected_models if results[model]['dev_accuracy'] is not None],
            [results[model]['test_accuracy'] for model in selected_models if results[model]['test_accuracy'] is not None],
            [model for model in selected_models if results[model]['train_accuracy'] is not None],
            f"{plot_output_dir}/grouped_train_dev_test_comparison.png"
        )

        return results
