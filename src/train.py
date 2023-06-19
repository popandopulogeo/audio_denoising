import os
from socket import gethostname
import wandb
import gc

import torch
import torch.nn as nn
from torch.distributed import is_initialized

from src.dataset import AudioDataset
from src.agents import LSSSDSepTrain, ddp_setup, ddp_destroy
from src.common import Config
from src.network import SilentIntervalDetection, NoiseEstimationNet, NoiseRemovalNet

def main():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])

    assert gpus_per_node <= torch.cuda.device_count()

    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    ddp_setup(world_size, rank)

    if rank == 0: 
        print(f"\nGroup initialized? {is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    torch.cuda.empty_cache()
    gc.collect()

    config = Config()

    components = {
        'SID' : SilentIntervalDetection(),
        'NE'  : NoiseEstimation(),
        'NR'  : NoiseRemoval()
    }

    optimizers = {key : torch.optim.Adam(params=components[key].parameters(), lr=config.base_lr) for key in components.keys()}
    schedulers = {key : torch.optim.lr_scheduler.OneCycleLR(optimizers[key], max_lr=2*config.base_lr, total_steps=config.n_epochs) for key in optimizers.keys()}
                                                    
    criterions = {
        'SID' : nn.BCELoss().to(local_rank),
        'NE'  : nn.MSELoss().to(local_rank),
        'NR'  : nn.MSELoss().to(local_rank)
    }

    train_loader, valid_loader = AudioDataset.get_dataloaders(config)
    
    if rank == 0:
        logger = wandb.init(
            project="Audio denoising",
            notes="LSSSD separated",
        )

        logger.save('src/*.py')
        logger.save('train.py')
    else:
        logger = None

    agent = LSSSDSepTrain(local_rank, components, optimizers, criterions, schedulers, logger, config)
    agent.train(train_loader, valid_loader)
    
    if rank == 0:
        logger.stop()

    ddp_destroy()

if __name__ == '__main__':
    main()
