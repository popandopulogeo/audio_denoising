import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.transform import *

import os
from abc import ABC
from pypesq import pesq
from sklearn.metrics import f1_score

def ddp_setup(world_size, rank):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_destroy():
    dist.destroy_process_group()

class TrainAgent(ABC):
    def __init__(self, local_rank, components, optimizers, criterions, schedulers, logger, config):
        self.local_rank = local_rank
        self.global_rank = int(os.environ["SLURM_PROCID"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.optimizers = optimizers
        self.criterions = criterions
        self.schedulers = schedulers
        self.components = components
        
        process_group = dist.new_group()
        for key in self.components.keys():
            self.components[key] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.components[key], process_group).to(local_rank)
            self.components[key] = DDP(self.components[key], device_ids=[local_rank], find_unused_parameters=True)

        self.logger = logger
        self.config = config
        if self.global_rank == 0:
            self.init_enviroment()

    def train(self, train_loader, valid_loader):
        for epoch in range(1, self.config.n_epochs + 1):
            ## Train
            self.stage = 'train'
            for key in self.components.keys():
                self.components[key].train()


            self.init_records()
            self.run_epoch(train_loader)
            self.reduce_records()
            if self.global_rank == 0:
                self.log_scalars()

            ## Validation
            self.stage = 'valid'
            for key in self.components.keys():
                self.components[key].eval()


            self.init_records()
            with torch.no_grad():
                self.run_epoch(valid_loader)
            self.reduce_records()
            if self.global_rank == 0:
                self.log_scalars()

            ## Creating snapshots
            if self.global_rank == 0:
                self.save_snapshot("LATEST.PTH")

                cur_metric = self.records["PESQ"].item() / self.n_samples.item()
                if cur_metric > self.best_metric:
                    self.save_snapshot("BEST.PTH")
                    self.best_metric = cur_metric

                self.current_epoch += 1

    @abstractmethod            
    def run_epoch(self):
        pass

    def compute_metrics(self, gt_stft, pred_stft, gt_mask, pred_mask):
        metrics = {
            'PESQ' : torch.tensor(0.0),
            'F1-score' : torch.tensor(0.0)
        }
        bacth_size = pred_stft.shape[0]

        for i in range(bacth_size):
            gt = istft2librosa(gt_stft[i], self.config.n_fft, self.config.hop_length, self.config.win_length)
            pred = istft2librosa(pred_stft[i], self.config.n_fft, self.config.hop_length, self.config.win_length)

            metrics['PESQ'] += torch.tensor(pesq(target, output, SR))
            metrics['F1-score'] += torch.tensor(f1_score(gt_mask[0], pred_mask[0], pos_label=1, average='binary', zero_division=1))
        return metrics

    def init_records(self):
        self.records = {f'{key}_LOSS' : torch.tensor(0.0) for key in self.criterions.keys()}    
        self.records['PESQ'] = torch.tensor(0.0)
        self.records['OVERALL_LOSS'] = torch.tensor(0.0)
        self.records['F1-score'] = torch.tensor(0.0)

        self.n_samples = torch.tensor(0)

    def update_records(self, record, n_samples) -> None:
        for key in self.records.keys():
            self.records[key] += record[key]
        self.n_samples += n_samples

    def reduce_records(self) -> None:
        for key in self.records.keys():
            self.records[key] = self.records[key].to(self.local_rank)
            dist.reduce(self.records[key], dst=0)
        
        self.n_samples = self.n_samples.to(self.local_rank)
        dist.reduce(self.n_samples, dst=0)

    def log_scalars(self) -> None:
        for key in self.records.keys(): 
            self.logger.log({f'metrics/{self.stage}/{key}' : torch.nan_to_num(self.records[key] / self.n_samples, -1).item()})

        for key in self.optimizers.keys():
            self.logger.log({f'metrics/{key}_lr' : self.optimizers[key].param_groups[0]['lr']})

    def save_snapshot(self, snapshot_name):
        """save shapshot during training for future restore"""
            
        snapshot_path = os.path.join(self.snapshots_root, snapshot_name)

        snapshot = {}
        for key in self.components.keys():
            snapshot[f'{key}_PARAMS'] = self.components[key].module.state_dict()

        for key in self.optimizers.keys():
            snapshot[f'{key}_OPTIMIZER'] = self.optimizers[key].state_dict()

        for key in self.schedulers.keys():
            if not self.schedulers[key] is None:
                snapshot[f'{key}_SCHEDULER'] = self.schedulers[key].state_dict()

        snapshot['CURRENT_EPOCH'] = self.current_epoch

        torch.save(snapshot, snapshot_path)
        self.logger[f"snapshots/{snapshot_name}"].upload(snapshot_path)
        print(f"Epoch {self.current_epoch} | Training snapshot saved at {snapshot_path}")

    def load_ckpt(self, snapshot_path):
        """load checkpoint from saved checkpoint"""

        snapshot = torch.load(snapshot_path, map_location=f"cuda:{self.local_rank}")

        for key in self.components.keys():
            self.components[key].module.load_state_dict(snapshot[f'{key}_PARAMS'])

        for key in self.optimizers.keys():
            self.optimizers[key].load_state_dict(snapshot[f'{key}_OPTIMIZER'])

        for key in self.schedulers.keys():
            if not self.schedulers[key] is None:
                self.schedulers[key].load_state_dict(snapshot[f'{key}_SCHEDULER'])

        self.current_epoch = snapshot['CURRENT_EPOCH']

        print(f"Checkpoint loaded from {snapshot_path}")

    def init_enviroment(self):

        self.snapshots_root = os.path.join('exp', self.logger.name)
        if not os.path.exists(self.snapshots_root):
            os.mkdir(self.snapshots_root)

        # Init counters

        self.current_epoch = 1
        self.best_metric = 0
  
class LSSSDAgent(TrainAgent):
    def __init__(self, local_rank, components, optimizers, criterions, schedulers, logger=None):
        super(LSSSDAgent, self).__init__(local_rank, components, optimizers, criterions, schedulers, logger)

    def run_epoch(self, loader):
        for batch in loader:

            mixed_stft = batch['mixed'].to(self.local_rank)
            true_sid_mask = batch['sid_mask'].to(self.local_rank)
            true_noise_stft = batch['noise'].to(self.local_rank)
            true_clean_stft = batch['clean'].to(self.local_rank)

            sid_mask = self.components['SID'](mixed_stft)
            sid_mask = torch.unsqueeze(sid_mask, 1)
            sid_mask = (sid_mask > 0.5).float()
            sid_mask_interpolated = F.interpolate(sid_mask, mixed_stft.shape[-1])
            sid_mask_interpolated = torch.unsqueeze(sid_mask_interpolated, 1)
            sid_mask = sid_mask.view(sid_mask.shape[0], -1), noise_mask, noise_stft

            noise_intervals = sid_mask_interpolated * x
            noise_stft = self.components['NE'](x, noise_intervals)
            noise_mask = self.components['NR'](x, noise_stft)
    
            clean_stft = batch_fast_icRM_sigmoid(mixed_stft, noise_mask)

            sid_loss = self.criterions['SID'](sid_mask, true_sid_mask)
            ne_loss = self.criterions['NE'](noise_stft, true_noise_stft),
            nr_loss = self.criterions['NR'](clean_stft, true_clean_stft)
            overall_loss = sid_loss + ne_loss + nr_loss

            if self.stage == 'train':
                self.optimizers['LSSSD'].zero_grad()
                overall_loss.backward()
                self.optimizers['LSSSD'].step()

            clean_stft = clean_stft.detach().cpu().numpy()
            true_clean_stft = true_clean_stft.detach().cpu().numpy()

            record = self.compute_metrics(true_clean_stft, clean_stft, true_sid_mask, sid_mask)
            record['SID_LOSS'] = sid_loss.item()
            record['NE_LOSS'] = ne_loss.item()
            record['NR_LOSS'] = nr_loss.item()
            record['OVERALL_LOSS'] = overall_loss.item()

            self.update_records(record, mixed_stft.shape[0])

        if self.stage == 'train':
            for key in self.schedulers.keys():
                if not self.schedulers[key] is None:
                    self.schedulers[key].step()

class DCRNAgent(TrainAgent):
    def __init__(self, local_rank, components, optimizers, criterions, schedulers, logger, config):
        super(DCRNAgent, self).__init__(local_rank, components, optimizers, criterions, schedulers, logger, config)
        self.reduction = lambda stft: torch.abs(stft[:,0]) + torch.abs(stft[:,1])

    def run_epoch(self, loader):
        for batch in loader:

            mixed_stft = batch['mixed'].to(self.config.device)
            true_clean_stft = batch['clean'].to(self.config.device)
            true_clean_stft = true_clean_stft.permute(0, 2, 3, 1)
            true_clean_audio = batch['audio'].to(self.config.device)

            clean_stft = self.components['DCRN'](mixed_stft)
            clean_stft = clean_stft.permute((0, 2, 3, 1))
            clean_audio = torch.istft(clean_stft, self.config.n_fft, self.config.hop_length, self.config.win_length) 

            Lt_loss = self.criterions['LT'](clean_audio, true_clean_audio[:,:clean_audio.shape[1]])
            Lf_loss = self.criterions['LF'](self.reduction(clean_stft), self.reduction(true_clean_stft))
            overall_loss = 0.4 * Lt_loss + 0.6 * Lf_loss

            if self.stage == 'train':
                self.optimizers['DCRN'].zero_grad()
                overall_loss.backward()
                self.optimizers['DCRN'].step()

            clean_audio = clean_audio.detach().cpu().numpy()
            true_clean_audio = true_clean_audio.detach().cpu().numpy()

            record = self.compute_metrics(true_clean_audio, clean_audio)
            record['LT_LOSS'] = Lt_loss.item()
            record['LF_LOSS'] = Lf_loss.item()
            record['OVERALL_LOSS'] = overall_loss.item()

            self.update_records(record, mixed_stft.shape[0])

        if self.stage == 'train':
            for key in self.schedulers.keys():
                if not self.schedulers[key] is None:
                    self.schedulers[key].step()

    def compute_metrics(self, gt_audio, pred_audio):
        metrics = {
            'PESQ' : torch.tensor(0.0),
        }
        bacth_size = pred_audio.shape[0]

        for i in range(bacth_size):
            metrics['PESQ'] += torch.tensor(pesq(gt_audio[i], pred_audio[i], self.config.sr))
        return metrics

class LSSSDSepAgent(TrainAgent):
    def __init__(self, local_rank, components, optimizers, criterions, schedulers, logger, config):
        super(LSSSDSepAgent, self).__init__(local_rank, components, optimizers, criterions, schedulers, logger, config)

    def run_epoch(self, loader):
        for batch in loader:

            mixed_stft = batch['mixed'].to(self.local_rank)
            true_sid_mask = batch['sid_mask'].to(self.local_rank)
            true_noise_stft = batch['noise'].to(self.local_rank)
            true_clean_stft = batch['clean'].to(self.local_rank)

            sid_mask = self.components['SID'](mixed_stft)
            sid_mask = torch.unsqueeze(sid_mask, 1)
            sid_mask = (sid_mask > 0.5).float()
            sid_mask_interpolated = F.interpolate(sid_mask, mixed_stft.shape[-1])
            sid_mask_interpolated = torch.unsqueeze(sid_mask_interpolated, 1)
            sid_mask = sid_mask.view(sid_mask.shape[0], -1), noise_mask, noise_stft

            noise_intervals = sid_mask_interpolated * x
            noise_stft = self.components['NE'](x, noise_intervals)
            noise_mask = self.components['NR'](x, noise_stft)
    
            clean_stft = batch_fast_icRM_sigmoid(mixed_stft, noise_mask)

            sid_loss = self.criterions['SID'](sid_mask, true_sid_mask)
            ne_loss = self.criterions['NE'](noise_stft, true_noise_stft),
            nr_loss = self.criterions['NR'](clean_stft, true_clean_stft)

            if self.stage == 'train':
                self.optimizers['SID'].zero_grad()
                sid_loss.backward()
                self.optimizers['SID'].step()

                self.optimizers['NE'].zero_grad()
                ne_loss.backward()
                self.optimizers['NE'].step()

                self.optimizers['NR'].zero_grad()
                nr_loss.backward()
                self.optimizers['NR'].step()

            clean_stft = clean_stft.detach().cpu().numpy()
            true_clean_stft = true_clean_stft.detach().cpu().numpy()

            record = self.compute_metrics(true_clean_stft, clean_stft, true_sid_mask, sid_mask)
            record['SID_LOSS'] = sid_loss.item()
            record['NE_LOSS'] = ne_loss.item()
            record['NR_LOSS'] = nr_loss.item()
            record['OVERALL_LOSS'] = sid_loss.item() + ne_loss.item() + nr_loss.item()

            self.update_records(record, mixed_stft.shape[0])

        if self.stage == 'train':
            for key in self.schedulers.keys():
                if not self.schedulers[key] is None:
                    self.schedulers[key].step()
            