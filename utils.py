import importlib
from const import ML_ALGORITHMS

def get_models(output_dir):
    models = {}
    for model_name, model_info in ML_ALGORITHMS.items():
        module = importlib.import_module(f"ml_algorithm.{model_info['file'][:-3]}")
        model_class = getattr(module, model_info['class'])
        models[model_name] = model_class(f"{output_dir}/params/{model_name}/{model_name.lower().replace(' ', '_')}_params.json")
    return models