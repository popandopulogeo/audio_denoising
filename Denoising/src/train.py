from denoising import DenoisingModel
from common import Config
import argparse

def main(args):

    config = Config()
    model = DenoisingModel(config)

    if args.cont:
        model.train(args.ckpt)
    else:
        model.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    args = parser.parse_args()

    main(args)
