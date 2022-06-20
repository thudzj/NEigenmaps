# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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

parser = argparse.ArgumentParser(description='DirectCLR Training')
parser.add_argument('--data', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=4.8, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', type=str, default='./logs/',
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=str, default='./logs/',
                    metavar='DIR', help='path to log directory')
parser.add_argument('--dim', default=360, type=int,
                    help="dimension of subvector sent to infoNCE")
parser.add_argument('--mode', type=str, default="baseline",
                    choices=["baseline", "simclr", "directclr", "single",
                             "neuralef"],
                    help="project type")
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--resume', type=str, default=None)

# parser.add_argument('--riemannian_projection', default=False, action='store_true')
# parser.add_argument('--neuralef_unloaded', default=False, action='store_true')
parser.add_argument('--max_grad_norm', default=None, type=float)
parser.add_argument('--kernel', default=0, type=int)
parser.add_argument('--opt', default='default', type=str)
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--proj_dim', default=[2048, 128], type=int, nargs='+')
parser.add_argument('--proj_bn', default=False, action='store_true')
parser.add_argument('--t', default=1, type=float)

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
    else:
        model = directCLR(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if args.opt == 'default':
        optimizer = LARS(model.parameters(),
                        lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=exclude_bias_and_norm,
                        lars_adaptation_filter=exclude_bias_and_norm)
    else:
        from timm.optim.optim_factory import create_optimizer_v2
        optimizer = create_optimizer_v2(model, opt=args.opt, lr=0,
                                        weight_decay=args.weight_decay,
                                        momentum=0.9,
                                        filter_bias_and_bn=True)

    # automatically resume from checkpoint if it exists
    if args.resume is not None:
        if args.resume == 'auto':
            if os.path.exists(os.path.join(args.checkpoint_dir, 'checkpoint.pth')):
                args.resume = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
            else:
                assert False
        ckpt = torch.load(args.resume, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    dataset = torchvision.datasets.ImageFolder(os.path.join(args.data, 'train'), Transform(args))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    if args.rank == 0:
        writer = SummaryWriter(log_dir = args.log_dir)
    else:
        writer = None

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        for step, ((y1, y2), labels) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, reg, cls_loss, acc = model.forward(y1, y2, labels)
                scaler.scale((loss + reg) * args.alpha + cls_loss).backward()
                if args.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if step % args.print_freq == 0:
                torch.distributed.reduce(acc.div_(args.world_size), 0)
                if args.rank == 0:
                    print(f'epoch={epoch}, step={step}, loss={loss.item():.6f},'
                          f' reg={reg.item():.6f}, cls_loss={cls_loss.item():.4f}'
                          f' acc={acc.item():.4f}, learning_rate={lr:.4f},'
                          f' t={F.softplus(model.module.logit_t).item():.4f}')
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(), reg=reg.item(),
                                 cls_loss=cls_loss.item(), acc=acc.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats), file=stats_file)

                if writer is not None:
                    writer.add_scalar('Loss/loss', loss.item(), step)
                    writer.add_scalar('Loss/reg', reg.item(), step)
                    writer.add_scalar('Loss/cls_loss', cls_loss.item(), step)
                    writer.add_scalar('Accuracy/train', acc.item(), step)
                    writer.add_scalar('Hparams/t', F.softplus(model.module.logit_t).item(), step)
                    writer.add_scalar('Hparams/lr', lr, step)

        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, os.path.join(args.checkpoint_dir, 'checkpoint.pth'))

    if args.rank == 0:
        # save final model
        torch.save(dict(backbone=model.module.backbone.state_dict(),
                        head=model.module.online_head.state_dict()),
                os.path.join(args.checkpoint_dir, 'final.pth'))


class directCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        self.online_head = nn.Linear(2048, 1000)

        if self.args.mode == "simclr":
            sizes = [2048, 2048, 128]
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i+1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[-1]))
            self.projector = nn.Sequential(*layers)
        elif self.args.mode == "single":
            self.projector = nn.Linear(2048, 128, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, y1, y2, labels):
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        if self.args.mode == "baseline":
            z1 = r1
            z2 = r2
        elif self.args.mode == "directclr":
            z1 = r1[:, :self.args.dim]
            z2 = r2[:, :self.args.dim]
        elif self.args.mode == "simclr" or self.args.mode == "single":
            z1 = self.projector(r1)
            z2 = self.projector(r2)


        loss = infoNCE(z1, z2) / 2 + infoNCE(z2, z1) / 2

        logits = self.online_head(r1.detach())
        cls_loss = F.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)
        return loss, torch.zeros_like(loss), cls_loss, acc


def infoNCE(nn, p, temperature=0.1):
    nn = F.normalize(nn, dim=1)
    p = F.normalize(p, dim=1)
    nn = gather_from_all(nn)
    p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = F.cross_entropy(logits, labels)
    return loss

class NeuralEFCLR(nn.Module):
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
                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
                # if args.proj_bn:
                #     layers.append(nn.BatchNorm1d(sizes[-1]))
                self.projector = nn.Sequential(*layers)
            elif len(args.proj_dim) == 1:
                self.projector = nn.Linear(2048, args.proj_dim[0], bias=True)
            else:
                self.projector = nn.Identity()
        if self.args.kernel == 4:
            self.logit_t = nn.Parameter(torch.tensor([args.t]).exp().div(1).log(), requires_grad=True)
        else:
            self.register_buffer('logit_t', torch.tensor([args.t]).exp().sub(1).log())

        if args.rank == 0:
            print(self.projector)

    def forward(self, y1, y2, labels):
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        z1 = self.projector(r1)
        z2 = self.projector(r2)

        psi1 = gather_from_all(z1)
        psi2 = gather_from_all(z2)

        B = psi1.shape[0] + psi2.shape[0]
        norm_ = (psi1.norm(dim=0) ** 2 + psi2.norm(dim=0) ** 2).div(B).sqrt().clamp(min=1e-6)


        psi1, psi2 = psi1.div(norm_), psi2.div(norm_)

        if self.args.kernel == 0:
            K_psi = torch.cat([psi2 - psi1, psi1 - psi2])
        elif self.args.kernel == 1:
            '''
            K:
                    x_1   x_2   x_3   x_4  x'_1  x'_2  x'_3  x'_4
            x_1  | -1/8  -1/8  -1/8  -1/8 1-1/8  -1/8  -1/8  -1/8
            x_2  | -1/8  -1/8  -1/8  -1/8  -1/8 1-1/8  -1/8  -1/8
            x_3  | -1/8  -1/8  -1/8  -1/8  -1/8  -1/8 1-1/8  -1/8
            x_4  | -1/8  -1/8  -1/8  -1/8  -1/8  -1/8  -1/8 1-1/8
            x'_1 |1-1/8  -1/8  -1/8  -1/8  -1/8  -1/8  -1/8  -1/8
            x'_2 | -1/8 1-1/8  -1/8  -1/8  -1/8  -1/8  -1/8  -1/8
            x'_3 | -1/8  -1/8 1-1/8  -1/8  -1/8  -1/8  -1/8  -1/8
            x'_4 | -1/8  -1/8  -1/8 1-1/8  -1/8  -1/8  -1/8  -1/8
            '''
            K_psi = torch.cat([psi2, psi1]) - (psi1.sum(0) + psi2.sum(0)) / B
        elif self.args.kernel == 2:
            K_psi = torch.cat([psi1 - psi2, psi2 - psi1])
        elif self.args.kernel == 3:
            '''
            K:
                    x_1   x_2   x_3   x_4  x'_1  x'_2  x'_3  x'_4
            x_1  |1-1/8  -1/8  -1/8  -1/8 1-1/8  -1/8  -1/8  -1/8
            x_2  | -1/8 1-1/8  -1/8  -1/8  -1/8 1-1/8  -1/8  -1/8
            x_3  | -1/8  -1/8 1-1/8  -1/8  -1/8  -1/8 1-1/8  -1/8
            x_4  | -1/8  -1/8  -1/8 1-1/8  -1/8  -1/8  -1/8 1-1/8
            x'_1 |1-1/8  -1/8  -1/8  -1/8 1-1/8  -1/8  -1/8  -1/8
            x'_2 | -1/8 1-1/8  -1/8  -1/8  -1/8 1-1/8  -1/8  -1/8
            x'_3 | -1/8  -1/8 1-1/8  -1/8  -1/8  -1/8 1-1/8  -1/8
            x'_4 | -1/8  -1/8  -1/8 1-1/8  -1/8  -1/8  -1/8 1-1/8
            '''
            K_psi = torch.cat([psi1 + psi2, psi2 + psi1]) - (psi1.sum(0) + psi2.sum(0)) / B
        elif self.args.kernel == 4:
            tmp = torch.tensor([[1., -1.], [-1., 1.]], device=self.logit_t.device)
            heat_kernel = torch.matrix_exp(-1 * F.softplus(-self.logit_t + (self.logit_t*2).detach()) * tmp)
            K_psi = torch.cat([psi1 * heat_kernel[0, 0] + psi2 * heat_kernel[0, 1],
                               psi1 * heat_kernel[0, 1] + psi2 * heat_kernel[0, 0]])

        psi_K_psi_diag = (torch.cat([psi1, psi2]) * K_psi).sum(0).view(-1, 1)
        psi_d_K_psi = torch.cat([psi1, psi2]).detach().T @ K_psi

        loss = - psi_K_psi_diag.sum() / (B ** 2)
        reg = (psi_d_K_psi ** 2 / psi_K_psi_diag.detach().abs().clamp_(min=1e-6)).triu(1).sum() / (B ** 2)

        logits = self.online_head(r1.detach())
        cls_loss = F.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)
        return loss, reg, cls_loss, acc

if __name__ == '__main__':
    main()
