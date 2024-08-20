import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class MLPlot:
    @staticmethod
    def save_plot(fig, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
        plt.close(fig)

    @classmethod
    def plot_grouped_accuracy_comparison(cls, train_accuracies, test_accuracies, model_names, filename):
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(model_names))
        width = 0.35

        train_bars = ax.bar(x - width/2, train_accuracies, width, label='Train', color='skyblue')
        test_bars = ax.bar(x + width/2, test_accuracies, width, label='Test', color='orange')

        ax.set_ylabel('Accuracy')
        ax.set_title('Model Train-Test Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()

        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=0)

        add_value_labels(train_bars)
        add_value_labels(test_bars)

        plt.tight_layout()
        cls.save_plot(fig, filename)

    @classmethod
    def plot_confusion_matrix(cls, cm, classes, filename):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        cls.save_plot(fig, filename)

    @classmethod
    def plot_best_accuracies_3d(cls, all_results, output_dir, features_dict):
        fig = plt.figure(figsize=(20, 15), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        algorithms = list(features_dict.keys())
        all_features = []
        for features in features_dict.values():
            all_features.extend(features)
        all_features = sorted(list(set(all_features)))

        xpos = np.arange(len(algorithms))
        ypos = np.arange(len(all_features))
        xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

        zpos = np.zeros((len(all_features), len(algorithms)))
        best_combo = {"algo": "", "feature": "", "model": "", "accuracy": 0}

        for i, algo in enumerate(algorithms):
            for j, feature in enumerate(all_features):
                if feature in all_results[algo]:
                    best_model = max(all_results[algo][feature], key=lambda m: all_results[algo][feature][m]['test_accuracy'])
                    accuracy = all_results[algo][feature][best_model]['test_accuracy']
                    zpos[j, i] = accuracy
                    if accuracy > best_combo["accuracy"]:
                        best_combo = {
                            "algo": algo,
                            "feature": feature,
                            "model": best_model,
                            "accuracy": accuracy
                        }

        dx = dy = 0.6
        dz = zpos.ravel()

        values = np.linspace(0.2, 1., xposM.ravel().shape[0])
        colors = cm.rainbow(values)

        bars = ax.bar3d(xposM.ravel(), yposM.ravel(), np.zeros_like(dz), dx, dy, dz, shade=True, color=colors)

        ax.set_xlabel('Feature Extraction Algorithms', fontsize=14, labelpad=20)
        ax.set_ylabel('Feature Extraction Methods', fontsize=14, labelpad=20)
        ax.set_zlabel('Test Accuracy', fontsize=14, labelpad=20)

        ax.set_xticks(xpos + dx/2)
        ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=12)

        ax.set_yticks(ypos + dy/2)
        ax.set_yticklabels(all_features, fontsize=8)

        ax.set_title('Test Accuracy for Different Feature Extraction Combinations', fontsize=16)

        # Highlight the best combination
        best_x = algorithms.index(best_combo["algo"])
        best_y = all_features.index(best_combo["feature"])
        ax.bar3d(best_x, best_y, 0, dx, dy, best_combo["accuracy"], color='red', alpha=1)

        # Add text annotation for the best combination
        ax.text(best_x + dx/2, best_y + dy / 2, best_combo["accuracy"] + 5 * 10e-3, 
                f"({best_combo['algo']}, {best_combo['feature']}, {best_combo['model']}, {best_combo['accuracy']:.4f})",
                color='red', fontweight='bold', ha='center', va='bottom', fontsize=10)

        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plot_path = f"{output_dir}/best_accuracies_3d_plot.png"
        cls.save_plot(fig, plot_path)

        print(f"3D plot of best accuracies saved to: {plot_path}")

        print(f"\nBest overall combination (The Ultimate Combo):")
        print(f"Feature Extraction Algorithm: {best_combo['algo']}")
        print(f"Feature Extraction Method: {best_combo['feature']}")
        print(f"ML Algorithm: {best_combo['model']}")
        print(f"Accuracy: {best_combo['accuracy']:.4f}")

        return best_combo