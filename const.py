FEATURES_DIC = {
    'time_domain': ['zcr', 'rms', 'envelope', 'temporal_moments', 'peak_to_peak'],
    'fourier': ['frequency_spectrum', 'cepstrum', 'magnitude_spectrum', 'phase_spectrum'],
    'short_fourier': ['mel', 'mfcc', 'chroma', 'spectral_contrast', 'zcr', 'spectral_rolloff', 'spectral_centroid', 'rms'],
    'wavelet': ['wavelet_coefficients', 'wavelet_packet', 'mra', 'wavelet_energy', 'wavelet_entropy'],
    'energy_intensity': ['pcm_RMSenergy', 'pcm_zcr', 'intensity', 'loudness']
}

ALL_MODELS = [
    'SVM',
    'Gradient Boosting',
    'K-Nearest Neighbors',
    'Logistic Regression',
    'Naive Bayes',
    'Decision Tree',
    'Multi-layer Perceptron'
]

LABELS_INPUT_DIR = "./test_audio/labels"
LABELS_OUTPUT_DIR = "./test_audio/labels/diseases_filtered"
LABELS_FILE_NAMES = [
    "dev_split",
    "train_split",
    "test_split"
]

DISEASE_COLUMNS = ['PHQ_Binary', 'PHQ_Score', 'PCL-C (PTSD)', 'PTSD Severity']
TARGET_COLUMNS = ["PTSD_class", "PHQ_class"]