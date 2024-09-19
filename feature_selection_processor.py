import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

class FeatureSelectionProcessor:
    def __init__(self):
        self.selected_features = None

    def fit(self, X, selected_features):
        self.selected_features = selected_features

    def transform(self, X):
        return X[self.selected_features]

    def fit_transform(self, X, n_features=10):
        _, X_selected = self.unsupervised_feature_selection(X, X.columns, n_features)
        self.fit(X, X_selected.columns)
        return X_selected

    def unsupervised_feature_selection(self, X, feature_names=None, n_features=10):
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]
        
        if feature_names is None:
            feature_names = X.columns
        
        feature_names_numeric = [name for name in feature_names if name in numeric_columns]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)

        pca = PCA(n_components=min(X_scaled.shape[1], 100))  # Limit to 100 components max
        X_pca = pca.fit_transform(X_scaled)

        feature_importance = np.sum(np.abs(pca.components_), axis=0)
        
        top_features_idx = np.argsort(feature_importance)[::-1][:n_features]
        selected_feature_names = np.array(feature_names_numeric)[top_features_idx]

        return selected_feature_names, X_numeric[selected_feature_names]

    def iterative_pca(self, X, variance_threshold=0.95):
        pca = PCA()
        pca.fit(X)

        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        optimal_n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1

        return optimal_n_components, cumulative_variance_ratio

    def apply_johnson_lindenstrauss(self, X, n_components):
        original_dim = X.shape[1]
        epsilon = np.sqrt(2 * np.log(original_dim) / n_components)
        random_matrix = np.random.normal(0, 1, (n_components, original_dim))
        random_matrix /= np.linalg.norm(random_matrix, axis=1)[:, np.newaxis]
        return (epsilon * X) @ random_matrix.T

    def find_optimal_clusters(self, X, max_clusters=10):
        silhouette_scores = []
        K = range(2, max_clusters + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))

        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        return optimal_k, silhouette_scores

    def restrict_to_largest_cluster(self, X, kmeans):
        cluster_sizes = np.bincount(kmeans.labels_)
        largest_cluster = np.argmax(cluster_sizes)
        mask = kmeans.labels_ == largest_cluster
        return X[mask], mask

    def select_optimal_features(self, X, y, n_features):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        initial_n_components, initial_cumulative_variance_ratio = self.iterative_pca(X_scaled)

        X_reduced = self.apply_johnson_lindenstrauss(X_scaled, initial_n_components)

        optimal_k, silhouette_scores = self.find_optimal_clusters(X_reduced)

        kmeans = KMeans(n_clusters=optimal_k)
        kmeans.fit(X_reduced)

        X_largest, largest_cluster_mask = self.restrict_to_largest_cluster(X_reduced, kmeans)

        final_n_components, final_cumulative_variance_ratio = self.iterative_pca(X_largest)

        final_pca = PCA(n_components=final_n_components)
        X_final_pca = final_pca.fit_transform(X_largest)

        feature_importance = np.sum(np.abs(final_pca.components_), axis=0)
        top_features = np.argsort(feature_importance)[::-1][:n_features]

        return top_features, optimal_k, final_n_components, silhouette_scores, final_cumulative_variance_ratio, largest_cluster_mask, X_reduced