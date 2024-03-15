import faulthandler

faulthandler.enable()

from pathlib import Path
import argparse
import os
import sys
import random
import subprocess
import time
import json
import numpy as np
import math

import torch
import torchvision
# from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from utils import *
from main import NeuralEFCLR, SpectralCLR

from retrieval import retrieval
from functools import partial

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--checkpoint-dir', type=str, default='./logs/',
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=str, default='./logs/',
                    metavar='DIR', help='path to log directory')
parser.add_argument('--mode', type=str, default="baseline",
                    choices=["neuralef", "spectral", "mrl"],
                    help="project type")
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--model', type=str, default='resnet50')

parser.add_argument('--proj_dim', default=[4096, 4096], type=int, nargs='+')
parser.add_argument('--no_proj_bn', default=False, action='store_true')
parser.add_argument('--momentum', default=0.99, type=float)

parser.add_argument('--coco_dir', default='../data/train2014', type=str)
parser.add_argument('--coco_db_path', default='../data/coco_DB.txt', type=str)
parser.add_argument('--coco_query_path', default='../data/coco_Query.txt', type=str)
parser.add_argument('--nuswide_dir', default='../data/nuswide_images', type=str)
parser.add_argument('--voc2012_dir', default='../data/', type=str)
parser.add_argument('--mirflickr_dir', default='../data/mirflickr', type=str)

parser.add_argument('--random_runs', default=None, type=int)
parser.add_argument('--normalize', default=None, type=str)
parser.add_argument('--pca', default=False, action='store_true')

# Dist
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-port', default='1234', type=str,
                    help='port used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

def main():
    args = parser.parse_args()

    if os.path.exists('/data/LargeData/Large/ImageNet'):
        args.data = '/data/LargeData/Large/ImageNet'
    elif os.path.exists('/home/LargeData/Large/ImageNet'):
        args.data = '/home/LargeData/Large/ImageNet'
    elif os.path.exists('/workspace/home/zhijie/ImageNet'):
        args.data = '/workspace/home/zhijie/ImageNet'
    elif os.path.exists('/home/data/ImageNet'):
        args.data = '/home/data/ImageNet'
    elif os.path.exists('/data/LargeData/Large/ImageNet'):
        args.data = '/data/LargeData/Large/ImageNet'

    args.proj_bn = not args.no_proj_bn
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank *= args.ngpus_per_node
    args.world_size *= args.ngpus_per_node
    args.dist_url = '{}:{}'.format(args.dist_url, args.dist_port)
    torch.multiprocessing.spawn(main_worker, (args,), nprocs=args.ngpus_per_node)

def main_worker(gpu, args):
    args.rank += gpu
    print(args.world_size, args.rank, args.dist_url)
    assert args.world_size == 1
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)
    args.log_dir = os.path.join(args.log_dir, args.name)
    if args.rank == 0:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        stats_file = open(args.checkpoint_dir + '/stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    if args.mode == 'neuralef':
        model = NeuralEFCLR(args).cuda(gpu)
    elif args.mode == 'spectral':
        model = SpectralCLR(args).cuda(gpu)
    elif args.mode == 'mrl':
        model = torchvision.models.resnet50(False)
        NESTING_LIST=[2**i for i in range(3, 12)]
        model.fc = MultiHeadNestedLinear(NESTING_LIST)
        apply_blurpool(model)	
        model.load_state_dict(get_ckpt(args.resume)) # Since our models have a torch DDP wrapper, we modify keys to exclude first 7 chars. 
        model.fc = nn.Identity()
        model = model.cuda(gpu)
    else:
        assert False

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # automatically resume from checkpoint if it exists
    if args.mode != 'mrl' and args.resume is not None:
        if args.resume == 'auto':
            if os.path.exists(os.path.join(args.checkpoint_dir, 'checkpoint.pth')):
                args.resume = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
            else:
                assert False
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)

    model.eval()

    if args.mode == 'neuralef' and args.normalize is not None and model.module.num_calls == 0:
        dataset = torchvision.datasets.ImageFolder(os.path.join(args.data, 'train'), Transform(args))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, num_workers=args.workers,
            pin_memory=True, sampler=None, shuffle=True)
        model.module.estimate_output_norm(loader, early_stop=None)

    if args.mode != 'mrl':
        inference_fn = partial(model.module.inference, normalize=args.normalize)
    else:
        inference_fn = model.module.forward
    # retrieval(args.data, '../data/imagenet_DB.txt', '../data/imagenet_Query.txt', model.device, inference_fn, 256, dname='imagenet', log_dir=args.log_dir, random_runs=args.random_runs)
    retrieval(args.nuswide_dir, '../data/nuswide_m_DB.txt', '../data/nuswide_m_Query.txt', model.device, inference_fn, 256, dname='nuswide_m', log_dir=args.log_dir, random_runs=args.random_runs, pca=args.pca)
    retrieval(args.voc2012_dir, '../data/voc2012_DB.txt', '../data/voc2012_Query.txt', model.device, inference_fn, 256, dname='voc2012', log_dir=args.log_dir, random_runs=args.random_runs, pca=args.pca)
    retrieval(args.mirflickr_dir, '../data/mirflickr_DB.txt', '../data/mirflickr_Query.txt', model.device, inference_fn, 256, dname='mirflickr', log_dir=args.log_dir, random_runs=args.random_runs, pca=args.pca)
    retrieval(args.coco_dir, args.coco_db_path, args.coco_query_path, model.device, inference_fn, 256, log_dir=args.log_dir, random_runs=args.random_runs, pca=args.pca)
    

if __name__ == '__main__':
    main()
