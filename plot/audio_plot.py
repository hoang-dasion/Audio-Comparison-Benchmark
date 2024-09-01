import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from const import METRICS_DIC

class AudioPlot:
    @staticmethod
    def plot_metric_comparison(results, metric, output_dir, algorithm):
        plot_dir = os.path.join(output_dir, algorithm, 'metric_comparisons')
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(15, 10))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
        for i, (feature_name, data) in enumerate(results.items()):
            durations = sorted([float(d) for d in data.keys() if d != 'full'])
            metric_values = [np.mean([file_data[metric] for file_data in data[str(d)].values()]) for d in durations]
            plt.plot(durations, metric_values, marker=markers[i % len(markers)], linestyle='-', linewidth=2, markersize=8, label=feature_name)
        
        plt.xlabel('Duration (s)', fontsize=12)
        plt.ylabel(METRICS_DIC[metric], fontsize=12)
        plt.title(f'{METRICS_DIC[metric]} Comparison - {algorithm.replace("_", " ").title()}', fontsize=14)
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        
        for i, (feature_name, data) in enumerate(results.items()):
            durations = sorted([float(d) for d in data.keys() if d != 'full'])
            metric_values = [np.mean([file_data[metric] for file_data in data[str(d)].values()]) for d in durations]
            for x, y in zip(durations, metric_values):
                plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        plt.savefig(os.path.join(plot_dir, f'{metric}_comparison.png'), bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_feature(feature, algorithm, feature_name, duration, output_dir, sr=22050, filename=None):
        plot_dir = os.path.join(output_dir, algorithm, feature_name, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        
        if algorithm == 'time_domain':
            times = np.linspace(0, len(feature) / sr, len(feature))
            plt.plot(times, feature)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
        elif algorithm in ['wavelet', 'short_fourier', 'fourier']:
            plt.imshow(np.atleast_2d(feature), aspect='auto', origin='lower')
            plt.colorbar(label='Magnitude')
            plt.xlabel('Coefficient' if algorithm == 'wavelet' else 'Frequency Bin')
            plt.ylabel('Scale' if algorithm == 'wavelet' else 'Time Frame')
        
        title = f'{algorithm.replace("_", " ").title()} - {feature_name} - Duration: {duration}s'
        if filename:
            title += f' - File: {filename}'
        plt.title(title)
        plt.tight_layout()
        
        if filename:
            plot_filename = f'{os.path.splitext(filename)[0]}_{feature_name}_{duration}.png'
        else:
            plot_filename = f'{feature_name}_{duration}.png'
        
        plt.savefig(os.path.join(plot_dir, plot_filename))
        plt.close()