import os
import numpy as np
import matplotlib.pyplot as plt

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