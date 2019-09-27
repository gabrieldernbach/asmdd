import os
import csv
import wave
import numpy as np
import librosa
import pandas as pd

def collect_file_paths(data_path = './dataset'):
    '''
    returns file names and the paths to the
    source wavs and target marks txt
    '''
    file_names = [name.strip('.wav') for name in os.listdir(data_path) if name.endswith('.wav')]
    # paths to audio source
    file_paths_wav = [os.path.join(data_path, file_name + '.wav') for file_name in file_names]
    # paths to target labels
    file_paths_txt = [os.path.join(data_path, file_name + '.txt') for file_name in file_names]
    print(len(file_names), 'wav files found in', data_path)
    return file_names, file_paths_wav, file_paths_txt

def infer_source_sampling_rate(file_paths_wav):
    '''
    return source_sampling rate and check if all files
    share the same sampling rate
    '''
    source_srs = []
    for file_path in file_paths_wav:
        with wave.open(file_path) as wavfile:
            source_srs += [wavfile.getframerate()]
    sample_rates_agree = len(set(source_srs)) == 1
    source_sr = source_srs[0]
    print('sample rate of first file is', source_sr)
    print('sample rates of other files agree =', sample_rates_agree)
    return source_sr

def mark_to_vec(mark_in_s, source_sr, len_sequence):
    '''
    return a vector of 0s and 1s according
    to the target marks given in seconds.

    mark_in_s are expected to contain a list of markers,
    each marker has a start in seconds mark[0]
    and a end at mark[1]

    return a vector of len_sequence that is
    activated at the source_sr sample rate time positions
    '''

    mark_in_samp = []
    durs = []
    for mark in mark_in_s:
        start = round(float(mark[0]) * source_sr)
        end = round(float(mark[1]) * source_sr)
        mark_in_samp.append([start,end])
        durs.append(float(mark[1]) - float(mark[0]))

    label_vec = np.zeros(len_sequence)
    for mark in mark_in_samp:
        label_vec[mark[0]:mark[1]] = 1

    return label_vec, durs

def build_dataframe(file_names,
                    file_paths_wav,
                    file_paths_txt,
                    source_sr,
                    target_sr):
    '''
    return loaded dataset as pandas dataframe object
    source_sr / target_sr determines resampling
    '''

    print('=== collecting data ===')
    pairs = []
    for idx in range(len(file_paths_wav)):
        # load and truncate sample rate [0]
        audio_content = librosa.core.load(file_paths_wav[idx], source_sr, mono=False)[0][0]

        # read file with target marks in second
        mark_in_s = list(csv.reader(open(file_paths_txt[idx],'r'),delimiter='\t'))
        # generate target vector from marks
        target_vec, durs = mark_to_vec(mark_in_s, source_sr, len(audio_content))

        # resample audio to lower sampling rate
        audio_content = librosa.core.resample(audio_content, source_sr, target_sr)
        # resample target vector and threshold to 0.5
        target_vector = (librosa.core.resample(target_vec, source_sr, target_sr) > 0.5) * 1

        # append to list
        pairs.append([file_names[idx], audio_content, target_vector, durs])
        print(f'load sample {idx}')

    data = pd.DataFrame(pairs, columns=['filename', 'audio_content', 'target_vector', 'durations_in_s'])
    print('dataframe built')
    return data


# DECIDING HYPTERPARAMETER downsampling
target_sr = 8000

file_names, file_paths_wav, file_paths_txt = collect_file_paths(data_path = './dataset')
source_sr = infer_source_sampling_rate(file_paths_wav)
data = build_dataframe(file_names, file_paths_wav, file_paths_txt, source_sr, target_sr)

print('safe file to "data.pkl"')
data.to_pickle('data.pkl')
print('success')
