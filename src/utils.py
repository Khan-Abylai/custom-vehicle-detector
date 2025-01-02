import os
import re
import torch
import torch.nn as nn
import numpy as np

from tools import transforms
from models.models import LPDetector


def get_latest_checkpoint(model_dir):
    files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    epoch_numbers = [int(f.split('_')[0]) for f in files]

    max_epoch = max(epoch_numbers)
    max_epoch_index = epoch_numbers.index(max_epoch)
    max_epoch_filename = os.path.join(model_dir, files[max_epoch_index])
    return max_epoch_filename


def get_model(input_size, gpu=0):
    model = LPDetector(input_size, cuda=gpu).cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    return model


def load_weights(model, model_directory, checkpoint_path, gpu=0):
    if gpu == 0 and not os.path.exists(model_directory):
        os.makedirs(model_directory)

    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{gpu}")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint['epoch']

    elif os.listdir(model_directory):
        checkpoint_path = get_latest_checkpoint(model_directory)
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{gpu}")
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    print(f'Loading from the checkpoint with epoch number {start_epoch}')

    return model, start_epoch + 1



def get_transforms():
    train_transforms = transforms.DualCompose([transforms.OneOf(
        [transforms.ImageOnly(transforms.GaussianBlur()), transforms.ImageOnly(transforms.AverageBlur()), ], prob=0.2),
        transforms.ImageOnly(transforms.RandomChannelPermute(prob=0.2)), #
        transforms.OneOf(
            [transforms.ImageOnly(transforms.ContrastNormalization()), transforms.ImageOnly(transforms.Add()), ],
            prob=0.2), transforms.ImageOnly(transforms.AdditiveGaussianNoise(prob=0.2)), transforms.OneOf(
            [transforms.ImageOnly(transforms.SaltAndPepper()), transforms.ImageOnly(transforms.Dropout()), ], prob=0.2),
        transforms.ImageOnly(transforms.RandomGrayScale(prob=0.2)), transforms.HorizontalFlip(prob=0.2),
        transforms.Rotate(limit=15, prob=0.2), transforms.ScaleDown(scale=0.5, prob=0.5),
        transforms.ImageOnly(transforms.Transpose()), transforms.Normalize(), transforms.BoxOnly(transforms.FillBox()),
        transforms.ToTensor()])

    val_transforms = transforms.DualCompose(
        [transforms.ImageOnly(transforms.Transpose()), transforms.Normalize(), transforms.BoxOnly(transforms.FillBox()),
            transforms.ToTensor()])

    return train_transforms, val_transforms
