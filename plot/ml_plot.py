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

    """
    @classmethod
    def plot_best_accuracies_3d(cls, all_results, output_dir, features_dict):
        fig = plt.figure(figsize=(20, 15), dpi=300)  # Increased figure size and DPI
        ax = fig.add_subplot(111, projection='3d')

        algorithms = list(all_results.keys())
        max_features = max(len(features) for features in features_dict.values())

        xpos = np.arange(len(algorithms))
        ypos = np.arange(max_features)
        xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

        zpos = np.zeros((max_features, len(algorithms)))
        dx = dy = 0.7

        best_combo = {"algo": "", "feature": "", "model": "", "accuracy": 0}

        for i, algo in enumerate(algorithms):
            for j, feature in enumerate(features_dict[algo]):
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

        zpos_flat = zpos.ravel()

        values = np.linspace(0.2, 1., xposM.ravel().shape[0])
        colors = cm.viridis(values)

        ax.bar3d(xposM.ravel(), yposM.ravel(), np.zeros_like(zpos_flat), dx, dy, zpos_flat, shade=True, color=colors, alpha=0.8)

        ax.set_xlabel('Feature Extraction Algorithms', fontsize=14, labelpad=20)
        ax.set_ylabel('Feature Extraction Methods', fontsize=14, labelpad=20)
        ax.set_zlabel('Test Accuracy', fontsize=14, labelpad=20)

        ax.set_xticks(xpos + dx/2)
        ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=12)

        max_feature_names = max(features_dict.values(), key=len)
        ax.set_yticks(ypos + dy/2)
        ax.set_yticklabels(max_feature_names, rotation=-20, ha='left', fontsize=10)

        ax.set_title('Test Accuracy for Different Feature Extraction Combinations', fontsize=18)

        # Highlight the best combination
        best_x = algorithms.index(best_combo["algo"])
        best_y = features_dict[best_combo["algo"]].index(best_combo["feature"])
        ax.bar3d(best_x, best_y, 0, dx, dy, best_combo["accuracy"], color='red', alpha=1)

        # Add text annotation for the best combination
        ax.text(best_x, best_y, best_combo["accuracy"], 
                f"Best: {best_combo['algo']}\n{best_combo['feature']}\n{best_combo['model']}\n{best_combo['accuracy']:.4f}", 
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
    """
    
    """
    @classmethod
    def plot_best_accuracies_3d(cls, all_results, output_dir, features_dict):
        fig = plt.figure(figsize=(24, 18), dpi=300)  # Increased figure size
        ax = fig.add_subplot(111, projection='3d')

        algorithms = list(all_results.keys())
        max_features = max(len(features) for features in features_dict.values())

        xpos = np.arange(len(algorithms))
        ypos = np.arange(max_features)
        xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

        zpos = np.zeros((max_features, len(algorithms)))
        dx = dy = 0.5  # Reduced bar width for more space between bars

        best_combo = {"algo": "", "feature": "", "model": "", "accuracy": 0}

        for i, algo in enumerate(algorithms):
            for j, feature in enumerate(features_dict[algo]):
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

        zpos_flat = zpos.ravel()

        values = np.linspace(0.2, 1., xposM.ravel().shape[0])
        colors = cm.viridis(values)

        ax.bar3d(xposM.ravel(), yposM.ravel(), np.zeros_like(zpos_flat), dx, dy, zpos_flat, shade=True, color=colors, alpha=0.8)

        ax.set_xlabel('Feature Extraction Algorithms', fontsize=16, labelpad=30)
        ax.set_ylabel('Feature Extraction Methods', fontsize=16, labelpad=30)
        ax.set_zlabel('Test Accuracy', fontsize=16, labelpad=30)

        ax.set_xticks(xpos + dx/2)
        ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=14)

        max_feature_names = max(features_dict.values(), key=len)
        ax.set_yticks(ypos + dy/2)
        ax.set_yticklabels(max_feature_names, rotation=-20, ha='left', fontsize=12)

        # Increase spacing between tick labels and axis
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', pad=10)

        ax.set_title('Test Accuracy for Different Feature Extraction Combinations', fontsize=20, pad=30)

        # Highlight the best combination
        best_x = algorithms.index(best_combo["algo"])
        best_y = features_dict[best_combo["algo"]].index(best_combo["feature"])
        ax.bar3d(best_x, best_y, 0, dx, dy, best_combo["accuracy"], color='red', alpha=1)

        # Add text annotation for the best combination
        ax.text(best_x, best_y, best_combo["accuracy"], 
                f"Best: {best_combo['algo']}\n{best_combo['feature']}\n{best_combo['model']}\n{best_combo['accuracy']:.4f}", 
                color='red', fontweight='bold', ha='center', va='bottom', fontsize=12)

        ax.view_init(elev=20, azim=45)

        # Adjust the layout to give more space to the axes
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        
        plot_path = f"{output_dir}/best_accuracies_3d_plot.png"
        cls.save_plot(fig, plot_path)

        print(f"3D plot of best accuracies saved to: {plot_path}")

        print(f"\nBest overall combination (The Ultimate Combo):")
        print(f"Feature Extraction Algorithm: {best_combo['algo']}")
        print(f"Feature Extraction Method: {best_combo['feature']}")
        print(f"ML Algorithm: {best_combo['model']}")
        print(f"Accuracy: {best_combo['accuracy']:.4f}")    
    """

    """
    @classmethod
    def plot_best_accuracies_3d(cls, all_results, output_dir, features_dict):
        fig = plt.figure(figsize=(20, 15), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        algorithms = list(features_dict.keys())
        top_combinations = {}

        for algo in algorithms:
            combinations = []
            for feature in features_dict[algo]:
                if feature in all_results[algo]:
                    for model, results in all_results[algo][feature].items():
                        accuracy = results['test_accuracy']
                        combinations.append((feature, model, accuracy))
            
            # Sort combinations and get top 3
            top_combinations[algo] = sorted(combinations, key=lambda x: x[2], reverse=True)[:3]

        x = np.arange(len(algorithms))
        y = np.arange(3)  # 3 bars for each algorithm
        
        dx = 0.2
        dy = 0.2
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))

        for i, algo in enumerate(algorithms):
            for j, (feature, model, accuracy) in enumerate(top_combinations[algo]):
                ax.bar3d(i, j, 0, dx, dy, accuracy, shade=True, color=colors[i], alpha=0.8)
                
                # Add text annotation for each combination
                ax.text(i + dx/2, j + dy/2, accuracy, 
                        f"{feature}\n{model}\n{accuracy:.4f}", 
                        color='black', fontweight='bold', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Feature Extraction Algorithms', fontsize=14, labelpad=20)
        ax.set_ylabel('Top 3 Combinations', fontsize=14, labelpad=20)
        ax.set_zlabel('Test Accuracy', fontsize=14, labelpad=20)

        ax.set_xticks(x + dx/2)
        ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=12)

        ax.set_yticks(y + dy/2)
        ax.set_yticklabels(['1st', '2nd', '3rd'], fontsize=10)

        ax.set_title('Top 3 Combinations for Each Feature Extraction Algorithm', fontsize=16)

        # Highlight the overall best combination
        best_algo = max(top_combinations, key=lambda k: top_combinations[k][0][2])
        best_combo = top_combinations[best_algo][0]
        best_index = algorithms.index(best_algo)
        ax.bar3d(best_index, 0, 0, dx, dy, best_combo[2], color='red', alpha=1)

        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plot_path = f"{output_dir}/top3_accuracies_3d_plot.png"
        cls.save_plot(fig, plot_path)

        print(f"3D plot of top 3 accuracies saved to: {plot_path}")

        print("\nTop 3 combinations for each algorithm:")
        for algo in algorithms:
            print(f"\n{algo}:")
            for i, (feature, model, accuracy) in enumerate(top_combinations[algo], 1):
                print(f"  {i}. {feature} - {model}: {accuracy:.4f}")

        print(f"\nBest overall combination (The Ultimate Combo):")
        print(f"Feature Extraction Algorithm: {best_algo}")
        print(f"Feature Extraction Method: {best_combo[0]}")
        print(f"ML Algorithm: {best_combo[1]}")
        print(f"Accuracy: {best_combo[2]:.4f}")
    """
    
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