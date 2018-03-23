import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from data import MNISTM
from models import Net


def main(args):
    dataset = MNISTM(train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=1, pin_memory=True)

    model = Net().cuda()
    model.load_state_dict(torch.load(args.MODEL_FILE))
    model.eval()

    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x = Variable(x, volatile=True).cuda()
        y_true = Variable(y_true, volatile=True).cuda()
        y_pred = model(x)

        total_accuracy += float((y_pred.max(1)[1] == y_true).float().mean())
    
    mean_accuracy = total_accuracy / len(dataloader)
    print(f'Accuracy on target data: {mean_accuracy:.4f}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Test a model on MNIST-M')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=256)
    args = arg_parser.parse_args()
    main(args)
