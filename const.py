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

COLOR_BARS = [
    "#FFA500", "#FFD700", "#FFFF00", "#9ACD32", "#32CD32", "#00FA9A",
    "#00CED1", "#1E90FF", "#0000FF", "#8A2BE2", "#9932CC", "#FF00FF",
    "#FF1493", "#FF69B4", "#DDA0DD", "#F0E68C", "#BDB76B", "#556B2F",
    "#008080", "#4682B4", "#483D8B", "#800080", "#FF4500"
]