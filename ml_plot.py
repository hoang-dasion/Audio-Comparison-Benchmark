import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

class MLPlot:
    @staticmethod
    def plot_confusion_matrix(cm, classes, filename):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_weighted_accuracies_3d(all_results, weighted_accuracies, output_dir, target):
        if not all_results or not weighted_accuracies:
            print(f"No data available to plot for target: {target}")
            return None

        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111, projection='3d')

        feature_combos = list(all_results[target].keys())
        algorithms = list(next(iter(all_results[target].values())).keys())

        xpos = np.arange(len(feature_combos))
        ypos = np.arange(len(algorithms))
        xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

        zpos = np.zeros((len(algorithms), len(feature_combos)))
        dx = dy = 0.8
        dz = zpos.ravel()

        for i, combo in enumerate(feature_combos):
            for j, algo in enumerate(algorithms):
                if algo in all_results[target][combo]:
                    accuracy = np.mean(weighted_accuracies[target][combo][algo])
                    zpos[j, i] = accuracy

        values = np.linspace(0.2, 1., xposM.ravel().shape[0])
        colors = cm.rainbow(values)

        ax.bar3d(xposM.ravel(), yposM.ravel(), np.zeros_like(dz), dx, dy, dz, 
                 shade=True, color=colors, alpha=0.8)

        ax.set_xlabel('Feature Combinations', fontsize=10, labelpad=20)
        ax.set_ylabel('ML Algorithms', fontsize=10, labelpad=20)
        ax.set_zlabel('Weighted Accuracy', fontsize=10, labelpad=20)

        ax.set_xticks(xpos + dx/2)
        ax.set_xticklabels(feature_combos, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(ypos + dy/2)
        ax.set_yticklabels(algorithms, fontsize=8)

        ax.set_title(f'Weighted Accuracies for Different Feature Combinations and ML Algorithms - {target}', fontsize=12)

        best_combo = max(
            ((combo, algo) 
             for combo in weighted_accuracies[target] 
             for algo in weighted_accuracies[target][combo]),
            key=lambda x: np.mean(weighted_accuracies[target][x[0]][x[1]])
        )

        best_feature_combo, best_algo = best_combo
        best_accuracy = np.mean(weighted_accuracies[target][best_feature_combo][best_algo])

        MLPlot._add_best_combo_annotation(ax, best_feature_combo, best_algo, best_accuracy, feature_combos, algorithms)

        ax.view_init(elev=20, azim=45)
        plot_path = os.path.join(output_dir, f"weighted_accuracies_3d_plot_{target}.png")
        plt.savefig(plot_path)
        plt.close(fig)

        print(f"3D plot of weighted accuracies for {target} saved to: {plot_path}\n")
        return best_combo

    @staticmethod
    def _add_best_combo_annotation(ax, best_feature_combo, best_algo, best_accuracy, feature_combos, algorithms):
        label_x = len(feature_combos) / 2
        label_y = len(algorithms) / 2
        label_z = ax.get_zlim()[1] * 1.2

        ax.text(label_x, label_y, label_z,
                f"Best Combination:\nFeature: {best_feature_combo}\nML Algo: {best_algo}\nWeighted Accuracy: {best_accuracy:.4f}",
                color='red', fontweight='bold', ha='center', va='bottom', fontsize=10)

        best_x = feature_combos.index(best_feature_combo)
        best_y = algorithms.index(best_algo)
        best_z = best_accuracy

        ax.text(label_x, label_y, label_z,
            f"Best Combination:\nFeature: {best_feature_combo}\nML Algo: {best_algo}\nWeighted Accuracy: {best_accuracy:.4f}",
            color='red', fontweight='bold', ha='center', va='bottom', fontsize=10, zorder=10)        

    @staticmethod
    def plot_stacked_accuracies_2d(all_results, output_dir, target):
        if not all_results[target]:
            print(f"No data available to plot for target: {target}")
            return None

        feature_combos = list(all_results[target].keys())
        algorithms = list(next(iter(all_results[target].values())).keys())

        if not feature_combos or not algorithms:
            print(f"Not enough data to create 2D stacked plot for target: {target}")
            return None

        fig, ax = plt.subplots(figsize=(15, 10))

        bottom = np.zeros(len(feature_combos))
        for algo in algorithms:
            accuracies = [all_results[target][combo][algo]['test_accuracy'] if algo in all_results[target][combo] else 0 for combo in feature_combos]
            ax.bar(feature_combos, accuracies, bottom=bottom, label=algo)
            bottom += accuracies

        ax.set_xlabel('Feature Combinations', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'Stacked Test Accuracies for Different Feature Combinations - {target}', fontsize=14)
        ax.legend(title='ML Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"stacked_accuracies_2d_plot_{target}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"2D stacked plot of accuracies for {target} saved to: {plot_path}\n")

        return max(
            ((combo, algo) 
            for combo in all_results[target] 
            for algo in all_results[target][combo]),
            key=lambda x: all_results[target][x[0]][x[1]]['test_accuracy']
        )

    @staticmethod
    def plot_weighted_model_performance_comparison(all_results, weighted_accuracies, output_dir, target, metric1, metric2):
        if not all_results[target] or not weighted_accuracies[target]:
            print(f"No data available to plot for target: {target}")
            return None

        feature_combos = list(all_results[target].keys())
        algorithms = list(next(iter(all_results[target].values())).keys())

        fig, ax = plt.subplots(figsize=(20, 10))

        metric1_values = []
        metric2_values = []
        labels = []

        for combo in feature_combos:
            for algo in algorithms:
                if algo in all_results[target][combo]:
                    metric1_values.append(all_results[target][combo][algo][metric1])
                    metric2_values.append(all_results[target][combo][algo][metric2])
                    labels.append(f"{combo}\n{algo}")

        x = np.arange(len(metric1_values))
        bar_width = 0.35
        
        # Plot bars for metric1 (test_f1) going up
        ax.bar(x - bar_width/2, metric1_values, bar_width, color='skyblue', alpha=0.7, label=metric1)
        
        # Plot bars for metric2 (train_f1) going down
        ax.bar(x + bar_width/2, [-v for v in metric2_values], bar_width, color='sandybrown', alpha=0.7, label=metric2)
        
        # Calculate and plot the difference
        difference = np.array(metric1_values) - np.array(metric2_values)
        ax.plot(x, difference, color='black', linewidth=2, label='Difference')
        
        # Plot the difference as bars overlapping the metric1 bars
        ax.bar(x - bar_width/2, difference, bar_width, color='lightgreen', alpha=0.5, label='Difference Bar')
        
        # Add a horizontal line at y=0.8
        ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% Threshold')

        ax.set_xlabel('Model Combinations', fontsize=10)
        ax.set_ylabel('Performance Metrics', fontsize=12)
        ax.set_title(f'{metric1} vs {metric2} Comparison - {target}', fontsize=14)
        ax.legend(loc='upper right')

        plt.xticks(x, labels, rotation=90, ha='right', fontsize=8)
        
        y_max = max(max(metric1_values), max(metric2_values), 0.8)
        y_min = min(min(-v for v in metric2_values), min(difference))
        ax.set_ylim(y_min - 0.1, y_max + 0.1)
        
        # Add value labels
        for i, (v1, v2, d) in enumerate(zip(metric1_values, metric2_values, difference)):
            ax.text(i - bar_width/2, v1, f'{v1:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
            ax.text(i + bar_width/2, -v2, f'{v2:.3f}', ha='center', va='top', fontsize=8, rotation=90)
            ax.text(i, d, f'{d:.3f}', ha='center', va='bottom' if d > 0 else 'top', fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{metric1}_vs_{metric2}_comparison_{target}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"{metric1} vs {metric2} comparison plot for {target} saved to: {plot_path}\n")

    @staticmethod
    def generate_plots(all_results, weighted_accuracies, f1_weighted_accuracies, output_dir):
        for target in all_results.keys():
            MLPlot.plot_stacked_accuracies_2d(all_results, output_dir, target)
            MLPlot.plot_weighted_accuracies_3d(all_results, weighted_accuracies, output_dir, target)
            MLPlot.plot_weighted_model_performance_comparison(all_results, weighted_accuracies, output_dir, target, 'test_accuracy', 'train_accuracy')
            MLPlot.plot_weighted_model_performance_comparison(all_results, weighted_accuracies, output_dir, target, 'test_accuracy', 'test_f1')
            MLPlot.plot_weighted_model_performance_comparison(all_results, f1_weighted_accuracies, output_dir, target, 'test_f1', 'train_f1')