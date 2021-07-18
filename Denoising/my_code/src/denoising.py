from os.path import join, exists
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from network import Model
from common import PHASE_TESTING, PHASE_TRAINING
from utils import TrainClock
from transform import *
from dataset import AudioDataset

class DenoisingModel(object):
    def __init__(self, config):
        self.config = config
        self.clock = TrainClock()

        self.net = Model()
        self.configurate_net()

        self.train_tb = SummaryWriter(join(self.config.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(join(self.config.log_dir, 'val.events'))

    def configurate_net(self):
        if torch.cuda.device_count() > 1:
            print('Multi-GPUs available')
            self.net = nn.DataParallel(self.net) 
        else:
            print('Single-GPU available')

        self.net.to(self.config.device)

        """set loss function"""
        self.loss = [nn.MSELoss(), nn.BCEWithLogitsLoss()]
        
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.config.lr_decay)

    def predict(self, input, data_type, audio_only=True):
        self.net = self.net.cuda()
        self.net.eval()

        with torch.no_grad():
            if data_type == 'stft':
                mixed_stft = torch.tensor(input.transpose((2, 0, 1)), dtype=torch.float32).cuda()
                mixed_stft = mixed_stft.view((-1,) + mixed_stft.shape)
                sid_mask_pred, noise_mask_pred, noise_pred = self.net(mixed_stft)
            elif data_type == 'raw':
                mixed_stft = fast_stft(input)
                mixed_stft = torch.tensor(mixed_stft.transpose((2, 0, 1)), dtype=torch.float32).cuda()
                mixed_stft = mixed_stft.view((-1,) + mixed_stft.shape)
                sid_mask_pred, noise_mask_pred, noise_pred = self.net(mixed_stft)

        sid_mask = sid_mask_pred.detach().cpu().numpy()[0]
        noise_stft = noise_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
        noise_mask = noise_mask_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
        mixed_stft = mixed_stft.detach().cpu().numpy()[0].transpose((1, 2, 0))

        clean_stft = fast_icRM_sigmoid(mixed_stft, noise_mask)

        if audio_only:
            return istft2librosa(clean_stft)
        else:
            return sid_mask, istft2librosa(clean_stft), istft2librosa(noise_stft)

    def train(self, ckpt=-1):
        if ckpt != -1:
            self.load_ckpt(ckpt)

        train_dataset = AudioDataset(PHASE_TRAINING, self.config)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.config.batch_size, 
                                  shuffle=True, 
                                  num_workers=self.config.num_workers,
                                  pin_memory=True, 
                                  worker_init_fn=np.random.seed())

        test_dataset = AudioDataset(PHASE_TESTING, self.config)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=self.config.batch_size, 
                                 num_workers=self.config.num_workers,
                                 pin_memory=True, 
                                 worker_init_fn=np.random.seed())

        losses_names = ['overall_loss', 'noise_loss', 'clean_loss', 'mask_loss']                         

        for epoch in range(self.clock.epoch, self.config.nr_epochs):

            train_loss_dict = dict.fromkeys(losses_names, [])
            test_loss_dict = dict.fromkeys(losses_names, [])

            #train part
            self.net.train()
            pbar = tqdm(train_loader)

            for i, data in enumerate(pbar):
                _, losses = self.forward(data)

                self.optimizer.zero_grad()
                losses['overall_loss'].backward()
                self.optimizer.step()

                pbar.set_description("EPOCH[{}][{}]".format(epoch, i))
                pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()})) #ordered dict
                
                for key, item in losses.items():
                    train_loss_dict[key].append(item.item())

                self.clock.tick()

            #validation part
            self.net.eval()
            pbar = tqdm(test_loader)

            with torch.no_grad():
                for data in pbar:
                    pbar.set_description("EVALUATION")
                    _, losses = self.forward(data)

                    for key, item in losses.items():
                        test_loss_dict[key].append(item.item())

            for key, item in train_loss_dict.items():
                self.train_tb.add_scalar(key, np.mean(item), global_step=self.clock.epoch)
            for key, item in test_loss_dict.items():  
                self.val_tb.add_scalar(key, np.mean(item), global_step=self.clock.epoch)
            self.train_tb.add_scalar("learning_rate", self.optimizer.param_groups[-1]['lr'], global_step=self.clock.epoch)
            
            self.scheduler.step(self.clock.epoch)
            self.clock.tock()

            if self.clock.epoch % self.config.save_frequency == 0:
                self.save_ckpt()
            self.save_ckpt('latest')

    def forward(self, data):
        mixed_stft = data['mixed'].to(self.config.device)
        noise_stft = data['noise'].to(self.config.device)
        clean_stft = data['clean'].to(self.config.device)
        sid_mask = data['sid_mask'].to(self.config.device)

        sid_mask_pred, noise_mask_pred, noise_pred = self.net(mixed_stft)

        clean_pred = batch_fast_icRM_sigmoid(mixed_stft, noise_mask_pred)

        losses = []
        losses.append(self.loss[0](noise_pred, noise_stft))
        losses.append(self.loss[0](clean_pred, clean_stft))
        losses.append(self.loss[1](sid_mask_pred, sid_mask))

        return clean_pred, {"noise_loss": losses[0], "clean_loss": losses[1], "mask_loss": losses[2], "overall_loss": sum(losses)}

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = join(self.config.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Checkpoint saved at {}\n".format(save_path))
        else:
            save_path = join(self.config.model_dir, "{}.pth".format(name))

        if isinstance(self.net, nn.DataParallel):
            torch.save({
                'clock': self.clock.save(),
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)
        else:
            torch.save({
                'clock': self.clock.save(),
                'model_state_dict': self.net.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)
        self.net.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = join(self.config.model_dir, "{}.pth".format(name))
        if not exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Checkpoint loaded from {}".format(load_path))

        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.clock.load(checkpoint['clock'])

    def load_weights(self, name):
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = join(self.config.project_root, 'models', "{}.pth".format(name))
        if not exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        self.net.load_state_dict(torch.load(load_path)['model_state_dict'])

    # def visualize_batch(self, data, mode, outputs=None, n=1):
    #     tb = self.train_tb if mode == PHASE_TRAINING else self.val_tb
    #     mixed_sig = data['mixed'][:n].numpy().transpose((0, 2, 3, 1))
    #     noise_sig = data['noise'][:n].numpy().transpose((0, 2, 3, 1))
    #     clean_sig = data['clean'][:n].numpy().transpose((0, 2, 3, 1))
    #     full_noise_sig = data['full_noise'][:n].numpy().transpose((0, 2, 3, 1))
    #     pred_noise_sig = outputs[0][:n].detach().cpu().numpy().transpose((0, 2, 3, 1))
    #     output_mask = outputs[1][:n].detach().cpu().numpy().transpose((0, 2, 3, 1))
    #     # groups = np.concatenate([mixed_sig, noise_sig, clean_sig, output_sig], axis=1)  # (n, 4, len)
    #     for i in range(n):
    #         output_sig = fast_icRM_sigmoid(mixed_sig[i], output_mask[i])
    #         # wavefrom = draw_waveform([mixed_sig[i], noise_sig[i], clean_sig[i], output_sig[i]])
    #         spectrum = draw_spectrum([fast_istft(mixed_sig[i]),
    #                                   fast_istft(noise_sig[i]),
    #                                   fast_istft(full_noise_sig[i]),
    #                                   fast_istft(pred_noise_sig[i]),
    #                                   fast_istft(clean_sig[i]),
    #                                   fast_istft(output_sig)])
    #         # wavefrom = wavefrom.transpose(2, 0, 1)[::-1]
    #         spectrum = spectrum.transpose(2, 0, 1)[::-1]

    #         # tb.add_image('waveform_{}'.format(i), wavefrom, global_step=self.clock.step)
    #         tb.add_image('spectrum_{}'.format(i), spectrum, global_step=self.clock.step)

    #         # tb.add_audio('mixed_{}'.format(i), mixed_sig[i:i+1], global_step=self.clock.step, sample_rate=self.sr)
    #         # tb.add_audio('noise_{}'.format(i), noise_sig[i:i+1], global_step=self.clock.step, sample_rate=self.sr)
    #         # tb.add_audio('clean_{}'.format(i), clean_sig[i:i+1], global_step=self.clock.step, sample_rate=self.sr)
    #         # tb.add_audio('output_{}'.format(i), output_sig[i:i+1], global_step=self.clock.step, sample_rate=self.sr)
