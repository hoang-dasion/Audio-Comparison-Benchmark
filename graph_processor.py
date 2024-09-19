import os
import numpy as np
import pandas as pd
import librosa
import networkx as nx
from tqdm import tqdm
import json
import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_process_audio(file_path, n_fft=2**11, hop_length=2**9, chunk_duration=5):
    """Load and process the audio file in small chunks."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        chunk_size = int(chunk_duration * sr)
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i+chunk_size]
            S = np.abs(librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length))
            yield chunk, S
    except Exception as e:
        logging.error(f"Error loading audio file {file_path}: {str(e)}")
        yield None, None

def create_visibility_graph(spectrum, max_nodes=200):
    """Create a visibility graph from spectrum."""
    if len(spectrum) > max_nodes:
        step = len(spectrum) // max_nodes
        spectrum = spectrum[::step]
    
    G = nx.Graph()
    G.add_nodes_from(range(len(spectrum)))
    for i in range(len(spectrum)):
        for j in range(i + 1, len(spectrum)):
            if all(spectrum[k] < spectrum[i] + (spectrum[j] - spectrum[i]) * (k - i) / (j - i) 
                   for k in range(i + 1, j)):
                G.add_edge(i, j)
    return G, spectrum

def extract_graph_features(G):
    """Extract all requested features from the graph."""
    try:
        avg_degree = np.mean([d for n, d in G.degree()])
        avg_clustering = nx.average_clustering(G)
        density = nx.density(G)
        transitivity = nx.transitivity(G)
        
        try:
            diameter = nx.diameter(G)
        except nx.NetworkXError:
            diameter = 0
        
        local_efficiency = nx.local_efficiency(G)
        global_efficiency = nx.global_efficiency(G)
        
        try:
            avg_shortest_path = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            avg_shortest_path = 0
        
        return {
            'avg_degree': avg_degree,
            'avg_clustering': avg_clustering,
            'density': density,
            'transitivity': transitivity,
            'diameter': diameter,
            'local_efficiency': local_efficiency,
            'global_efficiency': global_efficiency,
            'avg_shortest_path': avg_shortest_path
        }
    except Exception as e:
        logging.error(f"Error extracting graph features: {str(e)}")
        return None

def process_chunk(chunk, S):
    """Process a single chunk of the spectrogram."""
    try:
        spectrum = S.mean(axis=1)
        G, reduced_spectrum = create_visibility_graph(spectrum)
        features = extract_graph_features(G)
        return features, chunk, reduced_spectrum, G
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return None, None, None, None

def process_file(file_path, cache_dir):
    """Process a single audio file and extract graph features for all chunks."""
    try:
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{file_hash}.json")
        
        # Extract Participant_ID from filename
        participant_id = os.path.splitext(os.path.basename(file_path))[0]
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_features = json.load(f)
                return {'Participant_ID': participant_id, **cached_features}, None, None, None
        
        all_features = []
        chunks = []
        spectra = []
        graphs = []
        for i, (chunk, S) in enumerate(load_and_process_audio(file_path)):
            if chunk is None or S is None:
                continue
            chunk_features, chunk, spectrum, G = process_chunk(chunk, S)
            if chunk_features is not None:
                all_features.append(chunk_features)
                chunks.append(chunk)
                spectra.append(spectrum)
                graphs.append(G)
        
        if not all_features:
            logging.warning(f"No valid features extracted for {file_path}")
            return None, None, None, None
        
        avg_features = {
            key: np.mean([chunk[key] for chunk in all_features]) 
            for key in all_features[0].keys()
        }
        
        # Add Participant_ID as the first item in the dictionary
        final_features = {'Participant_ID': participant_id, **avg_features}
        
        with open(cache_file, 'w') as f:
            json.dump(final_features, f)
        
        return final_features, chunks, spectra, graphs
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None, None, None, None

def plot_file_features(file_path, chunks, spectra, graphs, output_dir):
    """Create plots for a single audio file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if not chunks or not spectra or not graphs:
            logging.warning(f"Not enough data to plot features for {file_path}")
            return
        
        chunk = chunks[0]
        spectrum = spectra[0]
        G = graphs[0]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
        ax1.plot(chunk)
        ax1.set_title('Audio Signal')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Amplitude')
        
        ax2.plot(spectrum)
        for edge in G.edges():
            ax2.plot([edge[0], edge[1]], [spectrum[edge[0]], spectrum[edge[1]]], 'r-', alpha=0.1)
        ax2.set_title('Connecting Peaks')
        ax2.set_xlabel('Frequency Bin')
        ax2.set_ylabel('Magnitude')
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax3, node_size=20, node_color='b', with_labels=False)
        ax3.set_title('Visibility Graph')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.basename(file_path)}_features.png"))
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting features for {file_path}: {str(e)}")

def process_file_wrapper(args):
    """Wrapper function for process_file to use with Pool.map()"""
    return process_file(*args)

def main():
    parser = argparse.ArgumentParser(description="Process audio files and generate graph features.")
    parser.add_argument("data_path", help="Path to the folder containing the data files")
    parser.add_argument("data_type", help="Type of data files to process (e.g., wav)")
    args = parser.parse_args()

    data_dir = args.data_path
    data_type = args.data_type
    output_file = f"{data_type}_graph_features.csv"
    cache_dir = "processing_cache"
    plot_dir = "./features_plot"
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f'.{data_type}')]
    logging.info(f"Found {len(audio_files)} {data_type} files")
    
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_file_wrapper, [(file, cache_dir) for file in audio_files]),
            total=len(audio_files),
            desc="Processing files"
        ))
    
    valid_results = [result for result in results if result[0] is not None]
    
    if not valid_results:
        logging.error("No valid results obtained. Exiting.")
        return
    
    features, chunks_list, spectra_list, graphs_list = zip(*valid_results)
    
    df = pd.DataFrame(features)
    
    # Ensure Participant_ID is the first column
    columns = ['Participant_ID'] + [col for col in df.columns if col != 'Participant_ID']
    df = df[columns]
    
    df.to_csv("./data/" + output_file, index=False)
    logging.info(f"All results saved to {output_file}")
    
    for file, chunks, spectra, graphs in zip(audio_files, chunks_list, spectra_list, graphs_list):
        if chunks and spectra and graphs:
            plot_file_features(file, chunks, spectra, graphs, plot_dir)
    
    logging.info(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    main()