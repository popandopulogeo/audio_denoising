import json
from os.path import join
import random
from pathlib import Path

import librosa
import numpy as np
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from tools import *
from transform import *
from common import PHASE_TESTING, PHASE_TRAINING

# datasets
##############################################################################
class AudioDataset(Dataset):
    def __init__(self, phase, config):
        print('========== DATASET CONSTRUCTION ==========')
        print('Initializing dataset...')
        super(AudioDataset, self).__init__()
        self.config = config

        self.data_root = join(self.config.data_root, phase)
        self.phase = phase
        self.dataset_json = join(self.config.data_root, phase + self.config.json_partial_name)

        print('Loading data...')
        with open(join(self.dataset_json), 'r') as fp:
            info = json.load(fp)
        self.dataset_path = info['dataset_path']
        self.num_files = info['num_files']
        self.files = info['files']
        print()

        print('Getting all noise files...')
        if phase == PHASE_TRAINING:
            self.noise_src = [f.resolve() for f in Path(self.config.noise_src_train).rglob('*.wav')]
        elif phase == PHASE_TESTING:
            self.noise_src = [f.resolve() for f in Path(self.config.noise_src_test).rglob('*.wav')]
        print()

        print('Loading all noise files...')
        self.noises = Parallel(n_jobs=-1, backend="multiprocessing")\
            (delayed(load_wav)(n, sr=self.config.sr) for n in tqdm(self.noise_src))
        print()

        print('Generating data items...')
        self.items = []
        if phase == PHASE_TRAINING:
            self.items = create_sample_list(self.files, 
                                            duration=self.config.duration,
                                            overlap=self.config.overlap)
        elif phase == PHASE_TESTING:
            self.items = create_sample_list(self.files, 
                                            percent_samples_selected=0.1, 
                                            duration=self.config.duration,
                                            overlap=self.config.overlap)
        self.num_samples = len(self.items)

        print('========== SUMMARY ==========')
        print('Mode:', phase)
        print('Dataset JSON:', self.dataset_json)
        print('Dataset path:', self.dataset_path)
        print('Num samples:', self.num_samples)
        print('Sample rate: {}'.format(self.config.sr))
        print('Clip length: {}'.format(self.config.duration))
        print('n_fft: {}'.format(self.config.n_fft))
        print('hop_length: {}'.format(self.config.hop_length))
        print('win_length: {}'.format(self.config.win_length))
        print()

    def __getitem__(self, index):
        item = self.items[index]
        # item[0]: audio clip index
        # item[1]: data start
        # item[2]: data end
        # item[3]: audio_path
        item = self.items[index]

        start = int(item[1] * SR)
        end = int(item[2] * SR)

        audio, _ = librosa.load(item[3], sr=self.config.sr)
        audio = audio[start:end]

        # compute mask
        sid_mask = np.zeros(self.config.time_bins)    
        sound = extrapolate_audio(np.abs(audio), self.config.time_bins)
        sound = sound.reshape(self.config.time_bins, -1)
        sid_mask = np.int_(np.mean(sound, axis=1) > self.config.mask_threshold)

        # read noise signal
        noise = random.choice(self.noises)
        snr = np.random.choice(self.config.SNRS, 1)[0]
        mixed_sig, clean_sig, noise_sig = add_noise_to_audio(audio, noise, snr)

        # stft
        mixed_sig_stft = fast_stft(mixed_sig, n_fft=self.config.n_fft, hop_length=self.config.hop_length, win_length=self.config.win_length)
        clean_sig_stft = fast_stft(clean_sig, n_fft=self.config.n_fft, hop_length=self.config.hop_length, win_length=self.config.win_length)
        noise_sig_stft = fast_stft(noise_sig, n_fft=self.config.n_fft, hop_length=self.config.hop_length, win_length=self.config.win_length)
        icrm = fast_cRM_sigmoid(clean_sig_stft, mixed_sig_stft)

        #to torch
        mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
        clean_sig_stft = torch.tensor(clean_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
        noise_sig_stft = torch.tensor(noise_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
        noise_mask = torch.tensor(icrm.transpose((2, 0, 1)), dtype=torch.float32)
        sid_mask = torch.tensor(sid_mask, dtype=torch.float32)


        return {
            "audio": audio,
            "mixed": mixed_sig_stft,
            "clean": clean_sig_stft,
            "noise": noise_sig_stft,
            "sid_mask": sid_mask,
            "noise_mask": noise_mask}

    def __len__(self):
        return len(self.items)

    @staticmethod
    def get_dataloaders(config):
        train_dataset = AudioDataset(PHASE_TRAINING, self.config)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.config.batch_size, 
                                  shuffle=True, 
                                  pin_memory=True,
                                  sampler=DistributedSampler(train_dataset,
                                                             num_replicas=int(os.environ["WORLD_SIZE"]),
                                                             rank=int(os.environ["SLURM_PROCID"])),
                                  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]) 
                                  )

        test_dataset = AudioDataset(PHASE_TESTING, self.config)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=self.config.batch_size, 
                                 pin_memory=True, 
                                 sampler=DistributedSampler(test_loader,
                                                            num_replicas=int(os.environ["WORLD_SIZE"]),
                                                            rank=int(os.environ["SLURM_PROCID"])),
                                 num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

        return train_loader, test_loader
