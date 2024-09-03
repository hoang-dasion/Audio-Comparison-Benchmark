AUDIO_ALGORITHMS = {
    'time_domain': {
        'class': 'TimeDomain',
        'file': 'time_domain.py',
        'features': ['zcr', 'rms', 'envelope', 'temporal_moments', 'peak_to_peak']
    },
    'fourier': {
        'class': 'Fourier',
        'file': 'fourier.py',
        'features': ['frequency_spectrum', 'cepstrum', 'magnitude_spectrum', 'phase_spectrum']
    },
    'short_fourier': {
        'class': 'ShortFourier',
        'file': 'short_fourier.py',
        'features': ['mel', 'mfcc', 'chroma', 'spectral_contrast', 'zcr', 'spectral_rolloff', 'spectral_centroid', 'rms']
    },
    'wavelet': {
        'class': 'Wavelet',
        'file': 'wavelet.py',
        'features': ['wavelet_coefficients', 'wavelet_packet', 'mra', 'wavelet_energy', 'wavelet_entropy']
    },
    'energy_intensity': {
        'class': 'EnergyIntensity',
        'file': 'energy_intensity.py',
        'features': ['pcm_RMSenergy', 'pcm_zcr', 'intensity', 'loudness']
    }
}

ML_ALGORITHMS = {
    'SVM': {
        'class': 'SVM',
        'file': 'svm.py'
    },
    'Random Forest': {
        'class': 'RandomForest',
        'file': 'random_forest.py'
    },
    'Gradient Boosting': {
        'class': 'GradientBoosting',
        'file': 'gradient_boosting.py'
    },
    'K-Nearest Neighbors': {
        'class': 'KNN',
        'file': 'knn.py'
    },
    'Logistic Regression': {
        'class': 'LogisticReg',
        'file': 'logistic_regression.py'
    },
    'Naive Bayes': {
        'class': 'NaiveBayes',
        'file': 'naive_bayes.py'
    },
    'Decision Tree': {
        'class': 'DecisionTree',
        'file': 'decision_tree.py'
    },
    'Multi-layer Perceptron': {
        'class': 'MLP',
        'file': 'mlp.py'
    }
}

LABELS_INPUT_DIR = "./test_audio/labels"
LABELS_OUTPUT_DIR = "./test_audio/labels/diseases_filtered"
LABELS_FILE_NAMES = [
    "dev_split",
    "train_split",
    "test_split"
]

DISEASE_COLUMNS = ['PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']
TARGET_COLUMNS = ["PTSD_class", "PHQ_class"]