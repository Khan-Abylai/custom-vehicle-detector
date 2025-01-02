import argparse
import os
import random
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch import optim
import torch.distributed as dist
from tqdm import tqdm
import torch.multiprocessing as mp

from dataset import CarDataset
import utils
import config

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format='[%(asctime)s:] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger('training')

def train(gpu, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.gpu_nums, rank=gpu)

    torch.cuda.set_device(gpu)

    model_directory = args.model_dir + args.model_name + '/'
    model = utils.get_model((args.img_w, args.img_h), gpu)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=0, eps=1e-6)
    model, start_epoch = utils.load_weights(model, model_directory, args.checkpoint, gpu)
    if args.lr is not None:
        optimizer.param_groups[0]['lr'] = args.lr

    train_transforms, val_transforms = utils.get_transforms()

    train_dataset = CarDataset(txt_files=config.train_txt_files, transforms=train_transforms,
                               size=(args.img_w, args.img_h), data_dir=args.data_dir, train=True)
    val_dataset = CarDataset(txt_files=config.valid_txt_files, transforms=val_transforms, size=(args.img_w, args.img_h),
                             data_dir=args.data_dir)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpu_nums, rank=gpu,
                                                                    shuffle=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.gpu_nums, rank=gpu,
                                                                  shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers // args.gpu_nums, sampler=train_sampler, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers // args.gpu_nums, sampler=val_sampler, drop_last=True)

    for epoch in range(start_epoch, args.num_epochs):

        train_mean_loss = 0
        n = 0
        i = 0

        train_num_car_boxes = 0
        train_correct_car_boxes = 0
        train_incorrect_car_boxes = 0

        progress_bar = train_dataloader
        if gpu == 0 and args.tqdm:
            progress_bar = tqdm(train_dataloader)

        for images, car_boxes in progress_bar:

            images = images.cuda(non_blocking=True)
            car_boxes = car_boxes.cuda(non_blocking=True)

            batch_size = images.shape[0]
            batch_info = model(images, car_boxes)
            batch_correct_car_boxes, batch_incorrect_car_boxes, batch_num_car_boxes, batch_car_loss = batch_info

            loss = batch_car_loss * 100

            train_current_loss = loss.item()

            loss.backward()
            if i % args.batch_multiplier == args.batch_multiplier - 1:
                optimizer.step()
                optimizer.zero_grad()

            train_correct_car_boxes += batch_correct_car_boxes
            train_incorrect_car_boxes += batch_incorrect_car_boxes

            train_num_car_boxes += batch_num_car_boxes
            train_recall_car_boxes = train_correct_car_boxes / train_num_car_boxes

            train_mean_loss = train_mean_loss * (n / (n + batch_size)) + train_current_loss * batch_size / (
                    n + batch_size)

            n += batch_size
            i += 1

            if gpu == 0:
                if args.tqdm:
                    progress_bar.set_description(
                        '{}/{} Cars {:.3f}/{} Loss: {:.3f}/{:.3f}'.format(epoch, args.num_epochs,
                                                                          train_recall_car_boxes,
                                                                          train_incorrect_car_boxes, train_current_loss,
                                                                          train_mean_loss))

                if args.logging:
                    logger.info(f'Train_loss={train_mean_loss};')

            if i == len(train_dataloader):

                val_mean_loss = 0
                n = 0

                val_num_car_boxes = 0
                val_correct_car_boxes = 0
                val_incorrect_car_boxes = 0

                model.eval()

                with torch.no_grad():
                    for images, car_boxes in val_dataloader:
                        images = images.cuda(non_blocking=True)
                        car_boxes = car_boxes.cuda(non_blocking=True)

                        batch_size = images.shape[0]
                        batch_info = model(images, car_boxes, validate=True)

                        batch_correct_car_boxes, batch_incorrect_car_boxes, batch_num_car_boxes, batch_car_loss = batch_info

                        loss = (batch_car_loss) * 100

                        val_current_loss = loss.item()

                        val_correct_car_boxes += batch_correct_car_boxes
                        val_incorrect_car_boxes += batch_incorrect_car_boxes

                        val_num_car_boxes += batch_num_car_boxes

                        val_mean_loss = val_mean_loss * (n / (n + batch_size)) + val_current_loss * batch_size / (
                                n + batch_size)

                        n += batch_size
                val_correct_car_boxes = torch.tensor(val_correct_car_boxes).cuda(non_blocking=True)
                val_incorrect_car_boxes = torch.tensor(val_incorrect_car_boxes).cuda(non_blocking=True)
                val_num_car_boxes = torch.tensor(val_num_car_boxes).cuda(non_blocking=True)
                dist.reduce(val_correct_car_boxes, 0)
                dist.reduce(val_incorrect_car_boxes, 0)
                dist.reduce(val_num_car_boxes, 0)
                val_recall_car_boxes = val_correct_car_boxes.item() / val_num_car_boxes.item()

                val_mean_loss = torch.tensor(val_mean_loss).cuda(non_blocking=True)
                dist.all_reduce(val_mean_loss)
                val_mean_loss = val_mean_loss.item() / args.gpu_nums

                train_epoch_description = '{}/{} TRAIN | Cars {:.3f}/{} Loss {:.3f}'
                val_epoch_description = ' VAL | Cars Recall {:.3f}/{} Val_loss {:.3f}, lr={}'

                epoch_description = train_epoch_description + val_epoch_description
                epoch_description = epoch_description.format(epoch, args.num_epochs, train_recall_car_boxes,
                                                             train_incorrect_car_boxes, train_mean_loss,
                                                             val_recall_car_boxes, val_incorrect_car_boxes,
                                                             val_mean_loss, optimizer.param_groups[0]['lr'])
                if gpu == 0:
                    if args.tqdm:
                        progress_bar.set_description(epoch_description)

                    if args.logging:
                        logger.info(epoch_description)

                    torch.save(
                        {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
                        model_directory + epoch_description.replace(' ', '_').replace('/', '_') + '.pth')

                model.train()
                scheduler.step(val_mean_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint weights')

    parser.add_argument('--img_w', type=int, default=640, help='image width')
    parser.add_argument('--img_h', type=int, default=640, help='image height')

    parser.add_argument('--lmdb', type=int, default=0, help='use lmdb')
    parser.add_argument('--tqdm', type=int, default=1, help='use tqdm')
    parser.add_argument('--gpu_nums', type=int, default=torch.cuda.device_count(), help='number of gpus')
    parser.add_argument('--anchors', type=int, default=0, help='use anchors')
    parser.add_argument('--calc_loss_weights', type=int, default=0, help='use loss balance by plate width')
    parser.add_argument('--batch_multiplier', type=int, default=1,
                        help='actual batch size = batch_size * batch_multiplier (use when cuda out of memory)')
    parser.add_argument('--logging', type=int, default=0, help='use logging')

    parser.add_argument('--model_name', type=str, default='car_detector', help='model name')
    parser.add_argument('--model_dir', type=str, default='/mnt/ssd/storage/weights/',
                        help='directory where model checkpoints are saved')
    parser.add_argument('--data_dir', type=str, default='/mnt/ssd/storage',
                        help='directory of data')
    parser.add_argument('--lmdb_name', type=str, default='train_data.lmdb')

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    random.seed(42)
    torch.manual_seed(42)
    mp.spawn(train, nprocs=args.gpu_nums, args=(args,))
