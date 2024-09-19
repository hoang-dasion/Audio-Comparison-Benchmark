import os
import time
import json
from functools import wraps

def milliseconds_to_hms(milliseconds):
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    return wrapper

def setup_output_directories():
    directories = [
        './output',
        './output/saved_models',
        './output/selected_features',
        './output/saved_scalers',
        './output/plots',
        './output/feature_selection_info'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)