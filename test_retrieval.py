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
from torch import nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from utils import *

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
                    choices=["neuralef", "spectral"],
                    help="project type")
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--model', type=str, default='resnet50')

parser.add_argument('--proj_dim', default=[2048, 128], type=int, nargs='+')
parser.add_argument('--no_proj_bn', default=False, action='store_true')

parser.add_argument('--coco_dir', default='/home/zhijie/data/train2014', type=str)
parser.add_argument('--coco_db_path', default='/home/zhijie/data/coco_DB.txt', type=str)
parser.add_argument('--coco_query_path', default='/home/zhijie/data/coco_Query.txt', type=str)
parser.add_argument('--nuswide_dir', default='/home/zhijie/data/nuswide_images', type=str)
parser.add_argument('--voc2012_dir', default='/home/zhijie/data/', type=str)
parser.add_argument('--mirflickr_dir', default='/home/zhijie/data/mirflickr', type=str)

parser.add_argument('--random_runs', default=None, type=int)
parser.add_argument('--bn', default=False, action='store_true')
parser.add_argument('--l2_normalize', default=False, action='store_true')

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

    args.proj_bn = not args.no_proj_bn
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank *= args.ngpus_per_node
    args.world_size *= args.ngpus_per_node
    args.dist_url = '{}:{}'.format(args.dist_url, args.dist_port)
    torch.multiprocessing.spawn(main_worker, (args,), nprocs=args.ngpus_per_node)

def main_worker(gpu, args):
    args.rank += gpu
    print(args.world_size, args.rank, args.dist_url)
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
    else:
        assert False

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # automatically resume from checkpoint if it exists
    if args.resume is not None:
        if args.resume == 'auto':
            if os.path.exists(os.path.join(args.checkpoint_dir, 'checkpoint.pth')):
                args.resume = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
            else:
                assert False
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])

    if args.rank == 0:
        model.eval()

        retrieval(args.nuswide_dir, '/home/zhijie/data/nuswide_m_DB.txt', '/home/zhijie/data/nuswide_m_Query.txt', model.device, model.module.inference, 256, dname='nuswide_m', log_dir=args.log_dir, random_runs=args.random_runs, bn=args.bn, l2_normalize=args.l2_normalize)
        retrieval(args.voc2012_dir, '/home/zhijie/data/voc2012_DB.txt', '/home/zhijie/data/voc2012_Query.txt', model.device, model.module.inference, 256, dname='voc2012', log_dir=args.log_dir, random_runs=args.random_runs, bn=args.bn, l2_normalize=args.l2_normalize)
        retrieval(args.mirflickr_dir, '/home/zhijie/data/mirflickr_DB.txt', '/home/zhijie/data/mirflickr_Query.txt', model.device, model.module.inference, 256, dname='mirflickr', log_dir=args.log_dir, random_runs=args.random_runs, bn=args.bn, l2_normalize=args.l2_normalize)
        # retrieval(args.data, '/home/zhijie/data/imagenet_DB.txt', '/home/zhijie/data/imagenet_Query.txt', model.device, model.module.inference, 256, dname='imagenet', log_dir=args.log_dir, random_runs=args.random_runs, bn=args.bn, l2_normalize=args.l2_normalize)
        retrieval(args.coco_dir, args.coco_db_path, args.coco_query_path, model.device, model.module.inference, 256, log_dir=args.log_dir, random_runs=args.random_runs, bn=args.bn, l2_normalize=args.l2_normalize)

        retrieval(args.nuswide_dir, '/home/zhijie/data/nuswide_m_DB.txt', '/home/zhijie/data/nuswide_m_Query.txt', model.device, partial(model.module.inference, binary=True), 256, dname='nuswide_m', log_dir=args.log_dir, random_runs=args.random_runs)
        retrieval(args.voc2012_dir, '/home/zhijie/data/voc2012_DB.txt', '/home/zhijie/data/voc2012_Query.txt', model.device, partial(model.module.inference, binary=True), 256, dname='voc2012', log_dir=args.log_dir, random_runs=args.random_runs)
        retrieval(args.mirflickr_dir, '/home/zhijie/data/mirflickr_DB.txt', '/home/zhijie/data/mirflickr_Query.txt', model.device, partial(model.module.inference, binary=True), 256, dname='mirflickr', log_dir=args.log_dir, random_runs=args.random_runs)
        # retrieval(args.data, '/home/zhijie/data/imagenet_DB.txt', '/home/zhijie/data/imagenet_Query.txt', model.device, partial(model.module.inference, binary=True), 256, dname='imagenet', log_dir=args.log_dir, random_runs=args.random_runs)
        retrieval(args.coco_dir, args.coco_db_path, args.coco_query_path, model.device, partial(model.module.inference, binary=True), 256, log_dir=args.log_dir, random_runs=args.random_runs)

    torch.distributed.barrier()

class NeuralEFCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = getattr(torchvision.models, args.model)(zero_init_residual=True)
        self.online_head = nn.Linear(self.backbone.fc.in_features, 1000)

        if args.proj_dim[0] == 0:
            self.projector = nn.Identity()
        else:
            if len(args.proj_dim) > 1:
                sizes = [self.backbone.fc.in_features,] + args.proj_dim
                layers = []
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                    if args.proj_bn:
                        layers.append(nn.BatchNorm1d(sizes[i+1]))
                    layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
                self.projector = nn.Sequential(*layers)
            elif len(args.proj_dim) == 1:
                self.projector = nn.Linear(2048, args.proj_dim[0], bias=True)

        self.backbone.fc = nn.Identity()

        if args.rank == 0 and hasattr(self, 'projector'):
            print(self.backbone)
            print(self.projector)

    def inference(self, y, k=None, binary=False):
        if binary:
            codes = hash_layer(self.projector(self.backbone(y)).sigmoid() - 0.5)
        else:
            codes = self.projector(self.backbone(y))
        if k == None:
            k = codes.shape[-1]
        return codes[...,:k]

class SpectralCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        self.online_head = nn.Linear(2048, 1000)

        if args.proj_dim[0] == 0:
            self.projector = nn.Identity()
        else:
            if len(args.proj_dim) > 1:
                sizes = [2048,] + args.proj_dim
                layers = []
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                    if args.proj_bn:
                        layers.append(nn.BatchNorm1d(sizes[i+1]))
                    layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
                if args.proj_bn:
                    layers.append(nn.BatchNorm1d(sizes[-1]))
                self.projector = nn.Sequential(*layers)
            elif len(args.proj_dim) == 1:
                self.projector = nn.Linear(2048, args.proj_dim[0], bias=True)

        if args.rank == 0 and hasattr(self, 'projector'):
            print(self.projector)

    def inference(self, y, k=None, binary=False):
        if binary:
            codes = hash_layer(self.projector(self.backbone(y)).sigmoid() - 0.5)
        else:
            codes = self.projector(self.backbone(y))
        if k == None:
            k = codes.shape[-1]
        return codes[...,:k]

class hash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def hash_layer(input):
    return hash.apply(input)

if __name__ == '__main__':
    main()
