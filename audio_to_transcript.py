import os
import sys
import csv
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import argparse
from tqdm import tqdm

def transcribe_audio_file(file_path):
    audio = AudioSegment.from_file(file_path)
    chunks = split_on_silence(audio, min_silence_len=700, silence_thresh=audio.dBFS-14, keep_silence=500)
    recognizer = sr.Recognizer()
    
    transcript = []
    start_time = 0
    
    for i, chunk in enumerate(tqdm(chunks, desc="Processing audio chunks")):

        chunk_filename = f"chunk{i}.wav"
        chunk.export(chunk_filename, format="wav")
        
        # Recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                stop_time = start_time + len(chunk) / 1000.0  # pydub works in milliseconds
                speaker = "Ellie" if i % 2 == 0 else "Participant"
                
                transcript.append({
                    "start_time": round(start_time, 3),
                    "stop_time": round(stop_time, 3),
                    "speaker": speaker,
                    "value": text
                })
                
                start_time = stop_time
        
        os.remove(chunk_filename)
    
    return transcript

def save_transcript_to_csv(transcript, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['start_time', 'stop_time', 'speaker', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        
        writer.writeheader()
        for row in transcript:
            writer.writerow(row)

def process_audio_files(data_path, file_type):
    for filename in os.listdir(data_path):
        if filename.endswith(f".{file_type}"):
            file_path = os.path.join(data_path, filename)
            print(f"Processing {filename}...")
            
            transcript = transcribe_audio_file(file_path)
            
            output_filename = f"{os.path.splitext(filename)[0]}_transcript.csv"
            output_path = os.path.join(data_path, output_filename)
            save_transcript_to_csv(transcript, output_path)
            
            print(f"Transcript saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Convert audio files to transcript CSV.")
    parser.add_argument("data_path", help="Path to the folder containing audio files")
    parser.add_argument("file_type", help="Type of audio files (e.g., m4a, wav)")
    
    args = parser.parse_args()
    
    process_audio_files(args.data_path, args.file_type)

if __name__ == "__main__":
    main()