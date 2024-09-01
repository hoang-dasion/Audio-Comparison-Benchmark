import os, json, shutil
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm
import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

## ENERGY BASED METHOD ##

def create_specs(in_path, out_path_qc, out_path_audio, target='PHQ_Binary', threshold=0.05):
    labels = pd.read_csv(os.path.join(in_path, 'labels.csv'))
    for file in os.listdir(in_path):
        if file.endswith('.wav'):
            print(f'Processing {file}')
            audio_path = os.path.join(in_path, file)
            y, sr = librosa.load(audio_path)

            # Compute the short-term energy
            frame_length = 2048
            hop_length = 512
            energy = np.array([
                sum(abs(y[i:i+frame_length]**2))
                for i in range(0, len(y), hop_length)
            ])

            # Normalize energy for better visualization
            energy = energy / np.max(energy)

            # Calculate time in minutes
            duration_in_seconds = len(y) / sr
            duration_in_minutes = duration_in_seconds / 60
            time_steps = np.linspace(0, duration_in_minutes, len(energy))

            if file.contains('_'):
                name = int(file.split('_')[0])
            else:
                name = file.split('.')[0]
            print(f'Participant: {name}')

            # Plot the energy with threshold
            plt.figure(figsize=(14, 5))
            plt.plot(time_steps, energy, label='Energy')
            plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Energy')
            plt.title(f'Energy of {name} with threshold')
            plt.legend()
            plt.savefig(os.path.join(out_path_qc, file.split('.')[0] + '.png'))
            plt.close()

            # Find segments where energy is above the threshold
            speech_segments = []
            start = None
            for i, e in enumerate(energy):
                if e > threshold and start is None:
                    start = i
                elif e < threshold and start is not None:
                    end = i
                    speech_segments.append((start, end))
                    start = None

            if start is not None:
                speech_segments.append((start, len(energy)))

            # Convert frames to time (in milliseconds for pydub)
            speech_segments = [(start * hop_length / sr * 1000, end * hop_length / sr * 1000) for start, end in speech_segments]

            # Filter segments to ensure minimum length of 5 seconds (5000 milliseconds)
            min_length_ms = 1000
            speech_segments = [(start, end) for start, end in speech_segments if (end - start) >= min_length_ms]

            # Load the audio using pydub
            audio = AudioSegment.from_wav(audio_path)

            # Create dirs for class labels
            try:
                class_ = labels.loc[labels['Participant_ID'] == name][target].values[0]
                print(f'Class: {class_}')
                if not os.path.exists(os.path.join(out_path_audio, str(class_), str(name))):
                    os.makedirs(os.path.join(out_path_audio, str(class_), str(name)))

                segment_path = os.path.join(out_path_audio, str(class_), str(name))

                # Save each segment as a separate file
                for i, (start, end) in enumerate(speech_segments):
                    segment = audio[start:end]
                    segment.export(os.path.join(segment_path, f'{name}_segment_{i}.wav'), format='wav')
            except IndexError:
                print(f'Participant {name} not found in labels.csv')

def det_threshold(in_path, filename='347.wav', start_time='00:00', end_time='02:30', start_speaking='00:30'):
    # Load the audio
    audio_path = os.path.join(in_path, filename)

    # Convert times in str format minutes:seconds to milliseconds
    start_time = int(start_time.split(':')[0]) * 60 * 1000 + int(start_time.split(':')[1]) * 1000
    end_time = int(end_time.split(':')[0]) * 60 * 1000 + int(end_time.split(':')[1]) * 1000
    start_speaking = int(start_speaking.split(':')[0]) * 60 * 1000 + int(start_speaking.split(':')[1]) * 1000

    # Get only the segment of interest
    audio = AudioSegment.from_wav(audio_path)
    audio = audio[start_time:end_time]

    # Convert AudioSegment to numpy array for librosa processing
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sr = audio.frame_rate

    # Compute the short-term energy
    frame_length = 2048
    hop_length = 512
    energy = np.array([
        sum(abs(samples[i:i+frame_length]**2))
        for i in range(0, len(samples), hop_length)
    ])

    # Normalize energy for better visualization
    energy = energy / np.max(energy)

    # Estimate threshold
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    threshold = mean_energy + 1.5 * std_energy

    # Calculate time steps in seconds
    time_steps = np.arange(0, len(energy)) * (hop_length / sr)

    # Identify segments where energy exceeds the threshold
    speaking_segments = []
    start_idx = None
    for i in range(len(energy)):
        if energy[i] > threshold and start_idx is None:
            start_idx = i
        elif energy[i] <= threshold and start_idx is not None:
            speaking_segments.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        speaking_segments.append((start_idx, len(energy)))

    # Plot the energy with mean and threshold
    plt.figure(figsize=(14, 5))
    plt.plot(time_steps, energy, label='Energy')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.axvline(x=(start_speaking - start_time) / 1000, color='b', linestyle='--', label='Begin speaking')

    # Highlight segments where energy exceeds the threshold
    for start, end in speaking_segments:
        plt.axvspan(time_steps[start], time_steps[end - 1], color='yellow', alpha=0.3)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy')
    plt.title(f'Energy of {filename.split(".")[0]} with Threshold energy = {threshold:.3f}')
    plt.legend()
    plt.savefig(os.path.join('daic', 'qc', filename.split('.')[0] + '_benchmark.png'))
    plt.show()

    return threshold
    
## USING TRANSCRIPTS ##
def classify_phq_score(phq_score):
    if phq_score <= 4:
        return 0  # Minimal depression
    elif phq_score <= 9:
        return 1  # Mild depression
    elif phq_score <= 14:
        return 2  # Moderate depression
    elif phq_score <= 19:
        return 3  # Moderately severe depression
    else:
        return 4  # Severe depression
    
def clasify_ptsd_score(pcl_5_score):
    if pcl_5_score < 20:
        return 0
    elif 20 <= pcl_5_score <= 30:
        return 1
    elif 31 <= pcl_5_score <= 40:
        return 2
    elif pcl_5_score > 40:
        return 3
    else:
        raise ValueError(f'Invalid PCL-5 score, {pcl_5_score}')
    
def process_file(in_path, out_path_audio, name, labels, target='PHQ_Binary', min_length_ms=500):
    '''helper function to process a single file and save participant segments as separate audio files'''
    # Load the audio file
    audio_path = os.path.join(in_path, name + '.wav')
    print(f'Processing {name}')

    try:
        class_ = labels.loc[labels['Participant_ID'] == int(name)][target].values[0]
        print(f'Class: {class_}')
        if not os.path.exists(os.path.join(out_path_audio, str(class_), str(name))):
            os.makedirs(os.path.join(out_path_audio, str(class_), str(name)))

        assert os.path.exists(os.path.join(out_path_audio, str(class_), str(name))), f'Path {os.path.join(out_path_audio, str(class_), str(name))} does not exist'

        segment_path = os.path.join(out_path_audio, str(class_), str(name))
        print(f'Saving segments to {segment_path}')

    except IndexError:
        print(f'Participant {name} not found in labels.csv')
        return None


    audio = AudioSegment.from_wav(audio_path)
    
    # Load the corresponding CSV file
    csv_path = os.path.join(in_path, name + '.csv')
    df = pd.read_csv(csv_path, sep='\t')
    
    # Initialize variables to store the start and stop times of participant segments
    participant_segments = []
    segment_start = None

    # Iterate over each row in the CSV
    for _, row in df.iterrows():
        if row['speaker'] == 'Participant' and segment_start is None:
            # Start a new segment
            segment_start = row['start_time']
        elif row['speaker'] == 'Ellie' and segment_start is not None:
            # End the current segment
            segment_end = row['start_time']
            participant_segments.append((segment_start, segment_end))
            segment_start = None

    # Process the last segment if it exists
    if segment_start is not None:
        participant_segments.append((segment_start, df['stop_time'].iloc[-1]))

    # Save each participant segment as a new audio file
    for i, (start, end) in enumerate(participant_segments):
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        if end_ms - start_ms > min_length_ms:
            segment = audio[start_ms:end_ms]
            segment_path_i = os.path.join(out_path_audio, str(class_), str(name), f'{name}_segment_{i}.wav')
            segment.export(segment_path_i, format='wav')
            print(f'Saved segment {i + 1} from {start} to {end} seconds as {segment_path}')

def process_all_files(in_path, out_path_audio, calc_segments=True):
    '''function to process all files in a given directory'''
    if calc_segments:
        labels = pd.read_csv(os.path.join('daic', 'labels.csv'))
        for file in tqdm(os.listdir(in_path)):
            if file.endswith('.wav'):
                filename = file.split('.')[0]
                print(f'Processing {filename}')
                process_file(in_path, out_path_audio, filename, labels)

    # copy all segments from 0 and 1 into one folder for ease of reference
    subdir_main = os.path.join(out_path_audio, '0+1')
    if not os.path.exists(subdir_main):
        os.makedirs(subdir_main)

    for subdir in os.listdir(os.path.join(out_path_audio, '0')):
        shutil.copytree(os.path.join(out_path_audio, '0', subdir), os.path.join(subdir_main, subdir))

    for subdir in os.listdir(os.path.join(out_path_audio, '1')):
        shutil.copytree(os.path.join(out_path_audio, '1', subdir), os.path.join(subdir_main, subdir))

def make_labels_complete(out_path_audio=os.path.join('daic', 'cut', '0+1'), labels_path=os.path.join('daic', 'labels.csv'), out_path=os.path.join('daic', 'labels_complete.csv')):
    labels = pd.read_csv(labels_path)
    labels_complete = pd.DataFrame()

    all_participants = os.listdir(out_path_audio)

    for participant in all_participants:
        print(f'Processing {participant}')
        labels_participant = labels.loc[labels['Participant_ID'] == int(participant)]
        phq = labels_participant['PHQ_Score'].values[0]
        ptsd = labels_participant['PTSD Severity'].values[0]
        phq_classified = classify_phq_score(phq)
        ptsd_classified = clasify_ptsd_score(ptsd)
        labels_participant['PHQ_class'] = phq_classified
        labels_participant['PTSD_class'] = ptsd_classified
        num_subsegments = len(os.listdir(os.path.join(out_path_audio, participant)))
        labels_participant['num_subseq'] = num_subsegments
        labels_complete = pd.concat([labels_complete, labels_participant])

    labels_complete.to_csv(out_path, index=False)

## visualization ##
def make_hist(out_path_audio):
    class_0_dir = os.path.join(out_path_audio, '0')
    class_1_dir = os.path.join(out_path_audio, '1')

    class_0_overall = len(os.listdir(class_0_dir))
    class_1_overall = len(os.listdir(class_1_dir))

    class_0 = [len(os.listdir(os.path.join(class_0_dir, participant))) for participant in os.listdir(class_0_dir)]
    class_1 = [len(os.listdir(os.path.join(class_1_dir, participant))) for participant in os.listdir(class_1_dir)]

    total_0 = sum(class_0) 
    total_1 = sum(class_1)

    plt.figure(figsize=(10, 5))
    plt.hist(class_0, alpha=0.5, label=f'Class 0, Total = {total_0}', color='blue')
    plt.hist(class_1, alpha=0.5, label=f'Class 1, Total = {total_1}', color='red')
    plt.xlabel('Number of segments')
    plt.ylabel('Count')
    plt.title(f'Total 0 participants= {class_0_overall}, Total 1 = {class_1_overall}')
    plt.legend()
    plt.savefig(os.path.join(out_path_audio, 'histogram.pdf'))

def make_hist_all(out_path_audio, save_path=os.path.join('daic', 'all_stats.json')):
    all_participants = os.path.join(out_path_audio, '0+1')

    # make a mapping of number of pariticpant segments by Gender,PHQ_Binary,PHQ_Score,PCL-C (PTSD),PTSD Severity
    males = []
    females = []
    phq_b0 = []
    phq_b1 = []
    phq_0 = []
    phq_1 = []
    phq_2 = []
    phq_3 = []
    phq_4 = []
    pcl_0 = []
    pcl_1 = []
    ptsd_0 = [] 
    ptsd_1 = []
    ptsd_2 = []
    ptsd_3 = []

    male_names = []
    female_names = []
    phq_b0_names = []
    phq_b1_names = []
    phq_0_names = []
    phq_1_names = []
    phq_2_names = []
    phq_3_names = []
    phq_4_names = []
    pcl_0_names = []
    pcl_1_names = []
    ptsd_0_names = []
    ptsd_1_names = []
    ptsd_2_names = []
    ptsd_3_names = []


    labels = pd.read_csv(os.path.join('daic', 'labels.csv'))

    for participant in os.listdir(all_participants):
        labels_participant = labels.loc[labels['Participant_ID'] == int(participant)]
        phq_b = labels_participant['PHQ_Binary'].values[0]
        phq = labels_participant['PHQ_Score'].values[0]
        pcl = labels_participant['PCL-C (PTSD)'].values[0]
        ptsd = labels_participant['PTSD Severity'].values[0]
        gender = labels_participant['Gender'].values[0]

        if phq_b == 0:
            phq_b0.append(len(os.listdir(os.path.join(all_participants, participant))))
            phq_b0_names.append(participant)
        else:
            phq_b1.append(len(os.listdir(os.path.join(all_participants, participant))))
            phq_b1_names.append(participant)

        phq_score = classify_phq_score(phq)
        if phq_score == 0:
            phq_0.append(len(os.listdir(os.path.join(all_participants, participant))))
            phq_0_names.append(participant)
        elif phq_score == 1:
            phq_1.append(len(os.listdir(os.path.join(all_participants, participant))))
            phq_1_names.append(participant)
        elif phq_score == 2:
            phq_2.append(len(os.listdir(os.path.join(all_participants, participant))))
            phq_2_names.append(participant)
        elif phq_score == 3:
            phq_3.append(len(os.listdir(os.path.join(all_participants, participant))))
            phq_3_names.append(participant)
        else:
            phq_4.append(len(os.listdir(os.path.join(all_participants, participant))))
            phq_4_names.append(participant)

        if pcl == 0:
            pcl_0.append(len(os.listdir(os.path.join(all_participants, participant))))
            pcl_0_names.append(participant)
        elif pcl == 1:
            pcl_1.append(len(os.listdir(os.path.join(all_participants, participant))))
            pcl_1_names.append(participant)

        ptsd_score = clasify_ptsd_score(ptsd)
        if ptsd_score == 0:
            ptsd_0.append(len(os.listdir(os.path.join(all_participants, participant))))
            ptsd_0_names.append(participant)
        elif ptsd_score == 1:
            ptsd_1.append(len(os.listdir(os.path.join(all_participants, participant))))
            ptsd_1_names.append(participant)
        elif ptsd_score == 2:
            ptsd_2.append(len(os.listdir(os.path.join(all_participants, participant))))
            ptsd_2_names.append(participant)
        elif ptsd_score == 3:
            ptsd_3.append(len(os.listdir(os.path.join(all_participants, participant))))
            ptsd_3_names.append(participant)

        if gender =='male':
            males.append(len(os.listdir(os.path.join(all_participants, participant))))
            male_names.append(participant)
        else:
            females.append(len(os.listdir(os.path.join(all_participants, participant))))
            female_names.append(participant)

    # save json
    stats =  {
        'males_names': male_names,
        'males_nums': males,
        'female_names': female_names,
        'female_nums': females,
        'phq_b0_names': phq_b0_names,
        'phq_b0_nums': phq_b0,
        'phq_b1_names': phq_b1_names,
        'phq_b1_nums': phq_b1,
        'phq_0_names': phq_0_names,
        'phq_0_nums': phq_0,
        'phq_1_names': phq_1_names,
        'phq_1_nums': phq_1,
        'phq_2_names': phq_2_names,
        'phq_2_nums': phq_2,
        'phq_3_names': phq_3_names,
        'phq_3_nums': phq_3,
        'phq_4_names': phq_4_names,
        'phq_4_nums': phq_4,
        'pcl_0_names': pcl_0_names,
        'pcl_0_nums': pcl_0,
        'pcl_1_names': pcl_1_names,
        'pcl_1_nums': pcl_1,
        'ptsd_0_names': ptsd_0_names,
        'ptsd_0_nums': ptsd_0,
        'ptsd_1_names': ptsd_1_names,
        'ptsd_1_nums': ptsd_1,
        'ptsd_2_names': ptsd_2_names,
        'ptsd_2_nums': ptsd_2,
        'ptsd_3_names': ptsd_3_names,
        'ptsd_3_nums': ptsd_3
    }

    with open(save_path, 'w') as f:
        json.dump(stats, f)

    plt.figure(figsize=(10, 5))
    plt.hist(males, alpha=0.5, label=f'Males (Total: {sum(males)})', color='blue', bins=10)
    plt.hist(females, alpha=0.5, label=f'Females (Total: {sum(females)})', color='red', bins=10)
    plt.xlabel('Number of segments')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join('daic', 'cut', 'gender.pdf'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(phq_b0, alpha=0.5, label=f'PHQ_Binary 0 (Total: {sum(phq_b0)})', color='blue', bins=10)
    plt.hist(phq_b1, alpha=0.5, label=f'PHQ_Binary 1 (Total: {sum(phq_b1)})', color='red', bins=10)
    plt.xlabel('Number of segments')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join('daic', 'cut', 'phq_binary.pdf'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(phq_0, alpha=0.5, label=f'PHQ 0 (Total: {sum(phq_0)})', color='blue', bins=10)
    plt.hist(phq_1, alpha=0.5, label=f'PHQ 1 (Total: {sum(phq_1)})', color='red', bins=10)
    plt.hist(phq_2, alpha=0.5, label=f'PHQ 2 (Total: {sum(phq_2)})', color='green', bins=10)
    plt.hist(phq_3, alpha=0.5, label=f'PHQ 3 (Total: {sum(phq_3)})', color='purple', bins=10)
    plt.hist(phq_4, alpha=0.5, label=f'PHQ 4 (Total: {sum(phq_4)})', color='orange', bins=10)
    plt.xlabel('Number of segments')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join('daic', 'cut', 'phq_score.pdf'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(pcl_0, alpha=0.5, label=f'PCL 0 (Total: {sum(pcl_0)})', color='blue', bins=10)
    plt.hist(pcl_1, alpha=0.5, label=f'PCL 1 (Total: {sum(pcl_1)})', color='red', bins=10)
    plt.xlabel('Number of segments')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join('daic', 'cut', 'pcl.pdf'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(ptsd_0, alpha=0.5, label=f'PTSD 0 (Total: {sum(ptsd_0)})', color='blue', bins=10)
    plt.hist(ptsd_1, alpha=0.5, label=f'PTSD 1 (Total: {sum(ptsd_1)})', color='red', bins=10)
    plt.hist(ptsd_2, alpha=0.5, label=f'PTSD 2 (Total: {sum(ptsd_2)})', color='green', bins=10)
    plt.hist(ptsd_3, alpha=0.5, label=f'PTSD 3 (Total: {sum(ptsd_3)})', color='purple', bins=10)
    plt.xlabel('Number of segments')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join('daic', 'cut', 'ptsd.pdf'))
    plt.close()


def recombination(spec_only=False):
    '''depricated, only used bc i had initially split the image files and os data into separate folders'''
    if not spec_only:
        os_complete = pd.DataFrame()
        for datatype in ['train', 'val', 'test']:
            os_part = pd.read_csv(os.path.join('daic', 'os', f'{datatype}', 'results_os.csv'))
            os_complete = pd.concat([os_complete, os_part])
        
        os_complete.to_csv(os.path.join('daic', 'os', 'results_os.csv'), index=False)
        print('done with os')

    # remove 'class' colume from results_os.csv
    os_complete = pd.read_csv(os.path.join('daic', 'os', 'results_os.csv'))
    os_complete = os_complete.drop(columns=['class'])
    os_complete.to_csv(os.path.join('daic', 'os', 'results_os.csv'), index=False)

    # copy all spectrograms from 0 and 1 into one folder for ease of reference using shutil
    # Target directory to collect all .png files
    target_dir = os.path.join('daic', 'spec', 'all')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Source directories
    data_types = ['train', 'val', 'test']
    class_types = ['0', '1']

    for data_type in data_types:
        for class_type in class_types:
            source_dir = os.path.join('daic', 'spec', data_type, class_type)
            for file in os.listdir(source_dir):
                    if file.endswith('.png'):
                        src_file = os.path.join(source_dir, file)
                        dst_file = os.path.join(target_dir, file)
                        print(f"Copying from {src_file} to {dst_file}")  # Debugging print statement
                        shutil.copy2(src_file, dst_file)
                        if os.path.exists(dst_file):
                            print(f"Successfully copied {dst_file}")  # Confirm successful copy
                        else:
                            print(f"Failed to copy {src_file}")  # Confirm failed copy
    print('Done')

def rejoin_segments(out_path_audio=os.path.join('daic', 'cut', '0+1'), out_path=os.path.join('daic', 'cut', 'joined')):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for participant in os.listdir(out_path_audio):
        print(f'Processing {participant}')
        joined = AudioSegment.empty()
        for segment in os.listdir(os.path.join(out_path_audio, participant)):
            segment_path = os.path.join(out_path_audio, participant, segment)
            segment_audio = AudioSegment.from_wav(segment_path)
            joined += segment_audio

        joined.export(os.path.join(out_path, f'{participant}.wav'), format='wav')

def process_rejoined_files(audio_path = os.path.join('daic', 'cut', 'joined'), out_path_os=os.path.join('daic', 'cut', 'joined_os'), clip_length=5):
    if not os.path.exists(out_path_os):
        os.makedirs(out_path_os)

    for file in tqdm(os.listdir(audio_path)):
        if file.endswith('.wav'):

            file_features = pd.DataFrame()
            filename_original = file.split('.')[0]

            # split file into clip_length second clips
            audio = AudioSegment.from_wav(os.path.join(audio_path, file))
            n_clips = len(audio) // (clip_length * 1000)
            for i in range(n_clips):
                clip = audio[i * clip_length * 1000: (i + 1) * clip_length * 1000]
                # clip.export(os.path.join(out_path_os, f'{file.split(".")[0]}_{i}.wav'), format='wav')
                # extract features
                features = smile.process_signal(clip.get_array_of_samples(), 16000)
                # filename = file.split('.')[0]+f'_{i}'
                # features['filename'] = filename
                file_features = pd.concat([file_features, features])

            features.to_csv(os.path.join(out_path_os, f'{filename_original}.csv'), index=False)

from scipy import signal

def create_specgram(audio_clip=None, temp_file=None, save_path=None, resize=True):
    '''create a spectrogram from a given audio file'''
    if audio_clip is None:
        y, sr = librosa.load(temp_file, sr=None)
    elif temp_file is None:
        y = audio_clip
        sr = 16000
    else:
        raise ValueError('Either audio_clip or temp_file must be provided')

    # Compute the spectrogram
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    if resize:
        S_db_resized = signal.resample(S_db, 100, axis=1)
        S_db_resized = signal.resample(S_db_resized, 100, axis=0)
        S_db = S_db_resized

    # Plot the spectrogram
    plt.figure()
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

    return S_db

def process_rejoined_files_spec(audio_path = os.path.join('daic', 'cut', 'joined'), out_path_spec=os.path.join('daic', 'cut', 'joined_spec'), clip_length=5):
    if not os.path.exists(out_path_spec):
        os.makedirs(out_path_spec)

    if not os.path.exists(os.path.join('daic', 'cut', 'temp')):
        os.makedirs(os.path.join('daic', 'cut', 'temp'))

    for file in tqdm(os.listdir(audio_path)):
        if file.endswith('.wav'):

            filename_original = file.split('.')[0]

            if not os.path.exists(os.path.join(out_path_spec, filename_original)):
                os.makedirs(os.path.join(out_path_spec, filename_original))

            # split file into clip_length second clips
            audio = AudioSegment.from_wav(os.path.join(audio_path, file))
            n_clips = len(audio) // (clip_length * 1000)
            for i in range(n_clips):
                clip = audio[i * clip_length * 1000: (i + 1) * clip_length * 1000]
                temp_file = os.path.join('daic', 'cut', 'temp', f'{filename_original}_{i}.wav')
                clip.export(temp_file, format='wav')
                # get spectrogram
                create_specgram(audio_clip=None, temp_file=temp_file, save_path=os.path.join(out_path_spec, filename_original, f'split_{i}.'), resize=True)

def split_files(target, gender, k_fold=None, leave_one_out=False):
    '''split into 70% train, 15% val, 15% test by number of files, but ensure that each participant is only in one set

    Args:
    - out_path_audio (str): path to the directory with the
    - label (str):  target
    - gender (str): 'male', 'female', 'all'
    - k_fold (int): number of folds for k-fold cross validation, if None, then do train-val-test split
    - leave_one_out (bool): if True, then split into leave-one-out cross validation
    '''
    labels_complete = pd.read_csv(os.path.join('daic', 'labels_complete.csv'))

    if gender != 'all':
        labels_complete = labels_complete.loc[labels_complete['Gender']==gender]

    labels_complete = labels_complete.sample(frac=1, random_state=47).reset_index(drop=True)

    def convert_to_native(data):
        if isinstance(data, np.int64):
            return int(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return [convert_to_native(item) for item in data]
        elif isinstance(data, dict):
            return {key: convert_to_native(value) for key, value in data.items()}
        else:
            return data
        
    class_unique = labels_complete[target].unique()

    if k_fold is None and not leave_one_out:

        # for each class of type target, need to ensure total split 70-15-15, using the column for target_num
        train = []
        val = []
        test = []
        train_labels = []
        val_labels = []
        test_labels = []
        for class_ in class_unique:
            class_df = labels_complete.loc[labels_complete[target]==class_]
            participants = class_df['Participant_ID'].unique()
            n_segments = class_df['num_subseq'].values
            total_segments = sum(n_segments)

            train_segments = 0
            val_segments = 0
            test_segments = 0
            for i, participant in enumerate(participants):
                additional_segments = n_segments[i] 
                if train_segments + additional_segments <= 0.7 * total_segments:
                    train.append(str(participant))
                    train_labels.append(class_)
                    train_segments += additional_segments
                elif val_segments + additional_segments <= 0.15 * total_segments:
                    val.append(str(participant))
                    val_labels.append(class_)
                    val_segments += additional_segments
                else:
                    test.append(str(participant))
                    test_labels.append(class_)
                    test_segments += additional_segments

            # print fractions per class
            print(f'Class {class_}: Train: {train_segments/total_segments}, Val: {val_segments/total_segments}, Test: {test_segments/total_segments}')

        # shuffle the splits consistently
        np.random.seed(47)
        idx = np.random.permutation(len(train))
        train = [train[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

        idx = np.random.permutation(len(val))
        val = [val[i] for i in idx]
        val_labels = [val_labels[i] for i in idx]

        idx = np.random.permutation(len(test))
        test = [test[i] for i in idx]
        test_labels = [test_labels[i] for i in idx]

        # save json
        splits = {
            'train': convert_to_native(train),
            'train_labels': convert_to_native(train_labels),
            'val': convert_to_native(val),
            'val_labels': convert_to_native(val_labels),
            'test': convert_to_native(test),
            'test_labels': convert_to_native(test_labels)
        }

        with open(os.path.join('daic', 'splits', f'splits_{target}_{gender}.json'), 'w') as f:
            json.dump(splits, f)
    
    elif k_fold is not None and not leave_one_out:
        # split data evenly into k folds
        folds_total = [{'participants': [], 'labels': [], 'segments': []} for _ in range(k_fold)]

        for class_ in class_unique:
            class_df = labels_complete.loc[labels_complete[target]==class_]
            participants = class_df['Participant_ID'].unique()
            n_segments = class_df['num_subseq'].values
            total_segments = sum(n_segments)
            n_samples_per_fold = total_segments // k_fold

            folds = []

            for i, participant in enumerate(participants):
                additional_segments = n_segments[i] 
                if len(folds) < k_fold:
                    folds.append({'participants': [str(participant)], 'labels': [class_], 'segments': [additional_segments]})
                else:
                    # find the fold with the least number of segments
                    min_fold = min(folds, key=lambda x: sum(x['segments']))
                    if sum(min_fold['segments']) + additional_segments <= n_samples_per_fold:
                        min_fold['participants'].append(str(participant))
                        min_fold['labels'].append(class_)
                        min_fold['segments'].append(additional_segments)
                    else:
                        print('Not enough space in any fold')
                        break

            for i, fold in enumerate(folds): # make json serializable
                folds_total[i]['participants'] += convert_to_native(fold['participants'])
                folds_total[i]['labels'] += convert_to_native(fold['labels'])
                folds_total[i]['segments'] += convert_to_native(fold['segments'])
            
        # save json
        with open(os.path.join('daic', 'splits',f'splits_{target}_{gender}_{k_fold}.json'), 'w') as f:
            json.dump(folds_total, f)

    elif leave_one_out:
        # randomly select one participant from each class to be in the test set, rest in train
        available_participants = labels_complete['Participant_ID'].unique()

        train = []
        test = []

        # randomly shuffle the participants
        np.random.seed(47)
        np.random.shuffle(available_participants)

        for participant in available_participants:
            # this is the removed participant
            remaining_participants = available_participants[available_participants != participant]
            remaining_labels = labels_complete.loc[labels_complete['Participant_ID'].isin(remaining_participants)]
            if gender != 'all':
                remaining_labels = remaining_labels.loc[remaining_labels['Gender']==gender]

            class_remaining = remaining_labels[target].values
            class_out = labels_complete.loc[labels_complete['Participant_ID']==participant][target].values[0]

            remaining_participants = [str(participant) for participant in remaining_participants]
            participant = str(participant)

            train.append({'participants': remaining_participants, 'labels': class_remaining})
            test.append({'participants': [participant], 'labels': [class_out]})

        # save json
        splits = {
            'train': convert_to_native(train),
            'test': convert_to_native(test)
        }

        with open(os.path.join('daic','splits', f'splits_{target}_{gender}_leave_one_out.json'), 'w') as f:
            json.dump(splits, f)
                
if __name__ == '__main__':
    in_path = os.path.join('daic', 'data')
    out_path_qc = os.path.join('daic', 'qc')
    out_path_audio = os.path.join('daic', 'cut')
    # create_specs(in_path, out_path_qc, out_path_audio)
    # det_threshold(in_path)
    # process_all_files(in_path, out_path_audio, calc_segments=True)
    # make_hist(out_path_audio)
    # split_files(out_path_audio)
    # make_hist_all(out_path_audio)
    # make_labels_complete()
    # recombination(spec_only=True)
    # split_files('phq_binary')

    # rejoin_segments()
    # process_rejoined_files()\
    process_rejoined_files_spec()

    # gender_ls = ['all', 'male', 'female']
    # target_ls = ['PHQ_Binary', 'PHQ_class','PTSD_class']
    # for target in target_ls:
    #     for gender in gender_ls:
    #         split_files(target, gender, k_fold=None, leave_one_out=True)