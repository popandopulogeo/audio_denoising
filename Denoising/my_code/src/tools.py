import librosa
import random
import math
import numpy as np

def create_sample_list(files, percent_samples_selected=None, duration=2, overlap=1):
    """Create a list of samples from indices"""
    all_choices = []
    for i, f in enumerate(files):
        f_sr = float(f['audio_sample_rate'])
        f_len = float(f['audio_samples'])
        f_duration = min(float(f['duration']), f_len/f_sr)

        if f_duration < duration:
            continue

        f_num_data = math.floor((f_duration - duration) / (duration - overlap)) + 1

        f_start_pos = np.arange(f_num_data) * (duration - overlap) 

        for x in f_start_pos:
            # construct choice (data) lists - list of tuples
            # item[0]: audio clip index
            # item[1]: data start
            # item[2]: data end
            # item[3]: audio_path
            choice = (i, x, x+duration, f['path'])
            all_choices.append(choice)

    print('Total available samples: ', len(all_choices))

    if percent_samples_selected is None:
        return all_choices

    if percent_samples_selected > 1:
        percent_samples_selected = 1
    elif percent_samples_selected < 0:
        percent_samples_selected = 0

    all_chosen_indices = sorted(np.random.choice(len(all_choices), int(len(all_choices)*percent_samples_selected), replace=False))
    result = [all_choices[i] for i in all_chosen_indices]
    return result

def load_wav(path, sr):
    return librosa.load(path, sr=sr)[0]

def power_of_signal(signal):
    return np.sum(np.abs(signal ** 2))

def add_noise_to_audio(audio, noise, snr, start_pos=None):
    # randomly load noise and randomly select an interval
    if start_pos is None:
        if len(noise) - len(audio) >= 1:
            start = random.randint(0, len(noise) - len(audio))
        elif len(noise) - len(audio) == 0:
            start = 0
        else:
            print('len(noise):', len(noise))
            print('len(audio):', len(audio))
            raise ValueError
        noise_cropped = noise[start:start+len(audio)]
    else:
        noise_cropped = noise[start_pos:start_pos+len(audio)]

    signal_power = power_of_signal(audio)
    pn = signal_power / np.power(10, snr / 10)

    if signal_power != 0:
        ratio = np.sqrt(power_of_signal(noise_cropped)) / np.sqrt(pn)
        if ratio != 0:
            noise_cropped = noise_cropped / ratio

    mixed = np.copy(audio)
    mixed += noise_cropped

    return mixed, audio, noise_cropped
    