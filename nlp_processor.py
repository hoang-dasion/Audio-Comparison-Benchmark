import os
import sys
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
import warnings
import argparse
import re

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

def extract_bert_features(text):
    """Extract BERT features from text."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except Exception as e:
        logging.error(f"Error extracting BERT features: {str(e)}")
        return None

def extract_participant_id(filename):
    """Extract Participant ID from filename."""
    # Pattern to match 'X#' at the start of the filename, where X is a letter and # is a number
    pattern = r'^([A-Za-z]\d+)_'
    
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    
    logging.warning(f"Could not extract Participant ID from filename: {filename}")
    return None

def process_file(file_path):
    """Process a single CSV file and extract features."""
    try:
        filename = os.path.basename(file_path)
        participant_id = extract_participant_id(filename)
        if participant_id is None:
            return None

        df = pd.read_csv(file_path, delimiter='\t')
        
        # Filter for Participant's speech only
        participant_text = df[df['speaker'] == 'Participant']['value'].str.cat(sep=' ')
        
        # If no participant text, use all text
        if not participant_text:
            participant_text = df['value'].str.cat(sep=' ')
            logging.warning(f"No participant text found in file {file_path}. Using all text.")

        if not participant_text:
            logging.warning(f"No text found in file {file_path}")
            return None
        
        bert_features = extract_bert_features(participant_text)
        
        if bert_features is not None:
            features = {'Participant_ID': participant_id}
            features.update({f'bert_{i}': v for i, v in enumerate(bert_features)})
            return features
        else:
            return None
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

def main(data_dir):
    output_file = "m4a_nlp_features.csv"
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logging.error(f"No CSV files found in {data_dir}")
        return
    
    logging.info(f"Processing {len(csv_files)} CSV files")
    
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in csv_files]
        for future in tqdm(as_completed(futures), total=len(csv_files), desc="Processing files"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    if not results:
        logging.error("No valid results obtained. Exiting.")
        return
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(data_dir, output_file), index=False)
    logging.info(f"All results saved to {output_file}")
    logging.info(f"Processed {len(df_results)} participants successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process transcript CSV files and extract NLP features.")
    parser.add_argument("data_path", help="Path to the folder containing transcript CSV files")
    args = parser.parse_args()
    
    main(args.data_path)