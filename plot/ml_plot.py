import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
import logging
from const import COLOR_BARS

class MLPlot:
    @staticmethod
    def plot_confusion_matrix(cm, classes, filename):
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_grouped_accuracy_comparison(train_accuracies, dev_accuracies, test_accuracies, model_names, filename):
        x = np.arange(len(model_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width, train_accuracies, width, label='Train')
        rects2 = ax.bar(x, dev_accuracies, width, label='Dev')
        rects3 = ax.bar(x + width, test_accuracies, width, label='Test')

        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)

        fig.tight_layout()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def save_plot(fig, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @classmethod
    def plot_best_accuracies_3d(cls, all_results, output_dir, target):
        fig = plt.figure(figsize=(20, 15), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        algorithms = list(all_results.keys())
        all_features = set()
        for algo_results in all_results.values():
            all_features.update(algo_results.keys())
        all_features = sorted(list(all_features))

        xpos = np.arange(len(algorithms))
        ypos = np.arange(len(all_features))
        xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

        zpos = np.zeros((len(all_features), len(algorithms)))
        best_combo = {"algo": "", "feature": "", "model": "", "accuracy": 0}

        dx = dy = 0.6
        dz = zpos.ravel()
        values = np.linspace(0.2, 1., xposM.ravel().shape[0])
        colors = cm.rainbow(values)

        for i, algo in enumerate(algorithms):
            for j, feature in enumerate(all_features):
                if feature in all_results[algo]:
                    valid_results = [x for x in all_results[algo][feature].items() if x[1]['test_accuracy'] is not None]
                    if valid_results:
                        best_model = max(valid_results, key=lambda x: x[1]['test_accuracy'])
                        accuracy = best_model[1]['test_accuracy']
                        zpos[j, i] = accuracy
                        if accuracy > best_combo["accuracy"]:
                            best_combo = {
                                "algo": algo,
                                "feature": feature,
                                "model": best_model[0],
                                "accuracy": accuracy
                            }

        bars = ax.bar3d(xposM.ravel(), yposM.ravel(), np.zeros_like(dz), dx, dy, dz, shade=True, color=colors)

        ax.set_xlabel('Feature Extraction Algorithms', fontsize=14, labelpad=20)
        ax.set_ylabel('Feature Extraction Methods', fontsize=14, labelpad=20)
        ax.set_zlabel('Test Accuracy', fontsize=14, labelpad=20)

        ax.set_xticks(xpos + dx/2)
        ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=12)

        ax.set_yticks(ypos + dy/2)
        ax.set_yticklabels(all_features, fontsize=8)

        ax.set_title(f'Test Accuracy for Different Feature Extraction Combinations - {target}', fontsize=16)

        # Highlight the best combination
        best_x = algorithms.index(best_combo["algo"])
        best_y = all_features.index(best_combo["feature"])
        ax.bar3d(best_x, best_y, 0, dx, dy, best_combo["accuracy"], color='red', alpha=1)

        ax.text(best_x + dx/2, best_y + dy/2, best_combo["accuracy"] + 0.05,
                f"Best: {best_combo['algo']},\n{best_combo['feature']},\n{best_combo['model']},\n{best_combo['accuracy']:.4f}\n\n\n\n",
                color='red', fontweight='bold', ha='center', va='bottom', fontsize=10)

        ax.view_init(elev=20, azim=45)

        plot_path = f"{output_dir}/best_accuracies_3d_plot_{target}.png"
        cls.save_plot(fig, plot_path)

        print(f"3D plot of best accuracies for {target} saved to: {plot_path}")
        print(f"\nBest overall combination for {target}:")
        print(f"Feature Extraction Algorithm: {best_combo['algo']}")
        print(f"Feature Extraction Method: {best_combo['feature']}")
        print(f"ML Algorithm: {best_combo['model']}")
        print(f"Accuracy: {best_combo['accuracy']:.4f}")

        return best_combo
