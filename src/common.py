from os.path import join

import torch
import utils

PHASE_TRAINING = 'training'
PHASE_TESTING = 'testing'

class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """
    def __init__(self):
        # general

        self.exp_name = "test"

        # experiment paths
        self.project_root = ""
        self.output_root = join(self.project_root, "model_output")
        self.exp_dir = join(self.output_root, self.exp_name)
        self.data_root = join(self.project_root, "data")

        self.noise_src_test = join(self.data_root, "noise_data_DEMAND", "noise_test")
        self.noise_src_train = join(self.data_root, "noise_data_DEMAND", "noise_train")

        self.log_dir = join(self.exp_dir, 'log')
        self.model_dir = join(self.exp_dir, 'model')
        utils.ensure_dirs([self.log_dir, self.model_dir])

        # dataset
        self.json_partial_name = '_TEDx1.json'
        self.csv_partial_name = '_TEDx1.csv'

        # data
        self.overlap = 1
        self.mask_threshold = 0.012
        self.sr = 16000
        self.duration = 2
        self.n_samples = self.duration*self.sr
        self.overlap_samples = self.overlap*self.sr

        self.SNRS = [-10, -7, -3, 0, 3, 7, 10]

        self.n_fft = 510
        self.hop_length = 158
        self.win_length = 400
        self.time_bins = 203

        # training configuration
        self.n_epochs = 150
        self.batch_size = 8 
        self.base_lr = 1e-4 


    def __repr__(self):
        return "epochs: {}\nbatch size: {}\nlr: {}\nworkers: {}\ndevice: {}\n".format(
            self.nr_epochs, self.batch_size, self.lr, self.num_workers, self.device
        )