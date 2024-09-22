from feature_selection_processor import FeatureSelectionProcessor
import numpy as np

def perform_feature_selection(X, y, n_features=10):
    feature_selector = FeatureSelectionProcessor()
    top_features, optimal_k, final_n_components, silhouette_scores, final_cumulative_variance_ratio, largest_cluster_mask, X_reduced = feature_selector.select_optimal_features(X, y, n_features)
    
    selected_feature_names = X.columns[top_features].tolist()
    X_selected = X.iloc[:, top_features]
    
    return selected_feature_names, X_selected, {
        'optimal_k': int(optimal_k),
        'final_n_components': int(final_n_components),
        'silhouette_scores': list(silhouette_scores),
        'final_cumulative_variance_ratio': list(final_cumulative_variance_ratio),
        'largest_cluster_mask': largest_cluster_mask.tolist(),
        'X_reduced': X_reduced.tolist()
    }