# const.py

FEATURES_DIC = {
    'time_domain': ['zcr', 'rms', 'envelope', 'temporal_moments', 'peak_to_peak'],
    'fourier': ['frequency_spectrum', 'cepstrum', 'magnitude_spectrum', 'phase_spectrum'],
    'short_fourier': ['mel', 'mfcc', 'chroma', 'spectral_contrast', 'zcr', 'spectral_rolloff', 'spectral_centroid', 'rms'],
    'wavelet': ['wavelet_coefficients', 'wavelet_packet', 'mra', 'wavelet_energy', 'wavelet_entropy']
}

METRICS_DIC = {
    'time': 'Computation Time (ms)',
    'memory': 'Memory Usage (MB)',
    'dimensionality': 'Feature Dimensionality'
}

ALL_MODELS = [
    'SVM',
    'Random Forest',
    'Gradient Boosting',
    'K-Nearest Neighbors',
    'Logistic Regression',
    'Naive Bayes',
    'Decision Tree',
    'Multi-layer Perceptron'
]

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