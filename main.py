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
parser.add_argument('--dim', default=360, type=int,
                    help="dimension of subvector sent to infoNCE")
parser.add_argument('--mode', type=str, default="baseline",
                    choices=["baseline", "simclr", "directclr", "single",
                             "neuralef", "neuralef-1lproj", "neuralef-proj"],
                    help="project type")
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--resume', type=str, default=None)

parser.add_argument('--riemannian_projection', default=False, action='store_true')
parser.add_argument('--neuralef_unloaded', default=False, action='store_true')
parser.add_argument('--max_grad_norm', default=None, type=float)
parser.add_argument('--kernel', default=0, type=int)


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
    if args.rank == 0:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        stats_file = open(args.checkpoint_dir + '/stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    if 'neuralef' in args.mode:
        model = NeuralEFCLR(args).cuda(gpu)
    else:
        model = directCLR(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    optimizer = LARS(model.parameters(),
                    lr=0, weight_decay=args.weight_decay,
                    weight_decay_filter=exclude_bias_and_norm,
                    lars_adaptation_filter=exclude_bias_and_norm)

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
                scaler.scale(loss + reg * 5 + cls_loss).backward()
                if args.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if step % args.print_freq == 0:
                torch.distributed.reduce(acc.div_(args.world_size), 0)
                if args.rank == 0:
                    print(f'epoch={epoch}, step={step}, loss={loss.item():.4f},'
                          f' reg={reg.item():.4f}, cls_loss={cls_loss.item():.4f}'
                          f' acc={acc.item():.4f}, learning_rate={lr}')
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(), reg=reg.item(),
                                 cls_loss=cls_loss.item(), acc=acc.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats), file=stats_file)
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
        cls_loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)
        return loss, torch.zeros_like(loss), cls_loss, acc


def infoNCE(nn, p, temperature=0.1):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    nn = gather_from_all(nn)
    p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

class NeuralEFCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        self.online_head = nn.Linear(2048, 1000)

        if self.args.mode == "neuralef-proj":
            sizes = [2048, 2048, 128]
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                # layers.append(nn.BatchNorm1d(sizes[i+1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            # layers.append(nn.BatchNorm1d(sizes[-1]))
            self.projector = nn.Sequential(*layers)
        elif self.args.mode == "neuralef-1lproj":
            self.projector = nn.Linear(2048, 128, bias=False)
        else:
            self.projector = nn.Identity()


        # '''
        # when multiplier=2, the sim_matrix is like
        #         x_1  x_2  x_3  x_4 x'_1 x'_2 x'_3 x'_4
        # x_1  |    1    0    0    0    1    0    0    0
        # x_2  |    0    1    0    0    0    1    0    0
        # x_3  |    0    0    1    0    0    0    1    0
        # x_4  |    0    0    0    1    0    0    0    1
        # x'_1 |    1    0    0    0    1    0    0    0
        # x'_2 |    0    1    0    0    0    1    0    0
        # x'_3 |    0    0    1    0    0    0    1    0
        # x'_4 |    0    0    0    1    0    0    0    1
        # '''
        # A = torch.tile(torch.eye(self.args.batch_size * args.world_size), (2, 2))
        #
        # print("the similarity matrix is like:")
        # print(torch.tile(torch.eye(4), (2, 2)))
        #
        # D = A.sum(-1).diag()
        # # L = D - A
        # # L_normalized = D.inverse().sqrt() @ L @ D.inverse().sqrt()
        #
        # self.K = D.inverse().sqrt() @ A @ D.inverse().sqrt() #L_normalized

    def forward(self, y1, y2, labels):
        # if self.args.rank == 0:
        #     print(y1[:2, :2])
        #     print(y2[:2, :2])

        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        # if self.args.rank == 0:
        #     print(r1[:6, :6])

        z1 = self.projector(r1)
        z2 = self.projector(r2)

        psi1 = gather_from_all(z1)
        psi2 = gather_from_all(z2)

        if 0:
            norm1_ = (psi1.norm(dim=0) ** 2).div(psi1.shape[0]).sqrt().clamp(min=1e-4)
            norm2_ = (psi2.norm(dim=0) ** 2).div(psi2.shape[0]).sqrt().clamp(min=1e-4)

            psi1, psi2 = psi1.div(norm1_), psi2.div(norm2_)

            psi_K_psi_diag = (psi1 * psi2).sum(0).view(-1, 1)
            psi2_d_K_psi1 = psi2.detach().T @ psi1
            psi1_d_K_psi2 = psi1.detach().T @ psi2

            loss = - psi_K_psi_diag.sum() * 2 / psi1.shape[0] * psi2.shape[0]

            #/ psi_K_psi_diag.detach()
            reg = (psi2_d_K_psi1 ** 2).triu(1).sum() \
                + (psi1_d_K_psi2 ** 2).triu(1).sum()
            reg /= psi1.shape[0] * psi2.shape[0]
        else:
            B = psi1.shape[0] + psi2.shape[0]
            norm_ = (psi1.norm(dim=0) ** 2 + psi2.norm(dim=0) ** 2).div(B).sqrt().clamp(min=1e-4)

            psi1, psi2 = psi1.div(norm_), psi2.div(norm_)

            if self.args.kernel == 0:
                K_psi = torch.cat([psi2 - psi1, psi1 - psi2])
            elif self.args.kernel == 1:
                K_psi = torch.cat([psi2, psi1]) - (psi1.sum(0) + psi2.sum(0)) / B
            elif self.args.kernel == 2:
                K_psi = torch.cat([psi1 - psi2, psi2 - psi1])
            elif self.args.kernel == 3:
                K_psi = torch.cat([psi1 + psi2, psi2 + psi1])

            psi_K_psi_diag = (torch.cat([psi1, psi2]) * K_psi).sum(0).view(-1, 1)
            psi_d_K_psi = torch.cat([psi1, psi2]).detach().T @ K_psi

            loss = - psi_K_psi_diag.sum()# / (B ** 2)

            reg = (psi_d_K_psi ** 2 / psi_K_psi_diag.detach().abs()).triu(1).sum() #
            # reg /= B ** 2

        # # estimate the neuralef grad
        # with torch.no_grad():
        #     # K = torch.tile(torch.eye(z1.shape[0]), (2, 2))
        #     # D = A.sum(-1).diag()
        #     # L = D - A
        #     # L_normalized = D.inverse().sqrt() @ L @ D.inverse().sqrt()
        #
        #     # self.K = D.inverse().sqrt() @ A @ D.inverse().sqrt()
        #
        #     # K_psi = K @ psi
        #     # if self.args.rank == 0:
        #     #     print(psi[:3, :3])
        #     with torch.cuda.amp.autocast(False):
        #         psi1_f, psi2_f = psi1.float(), psi2.float()
        #         K_psi2 = psi2_f
        #         psi1_K_psi2 = psi1_f.T @ K_psi
        #         if self.args.neuralef_unloaded:
        #             grad = K_psi - psi.float() @ psi_K_psi.tril(diagonal=-1).T / z.shape[0] # not sure, to check
        #         else:
        #             mask = torch.eye(z.shape[1], device=z.device) - \
        #                 (psi_K_psi / psi_K_psi.diag().clamp(min=1e-6)).tril(diagonal=-1).T
        #             grad = K_psi @ mask
        #
        #         # the trick in eigengame paper
        #         if self.args.riemannian_projection:
        #             grad.sub_((psi.float()*grad).sum(0) * psi.float() / z.shape[0])
        #
        #         # the scaling may be unnecessary
        #         grad *= - 1 #2 / z.shape[0]**2
        #
        #         # if self.args.max_grad_norm is not None:
        #         #     clip_coef = self.args.max_grad_norm / grad.norm(dim=0).clamp(min=1e-6)
        #         #     grad.mul_(clip_coef)
        #
        #         # grad = grad.type_as(psi)
        #
        #         # if self.args.rank == 0:
        #         #     print(grad[:10, :10], torch.isnan(grad).float().sum())
        #
        #         # it is a pseudo loss whose gradient w.r.t. psi is the `grad'
        #         loss = (psi.float() * grad).sum()

        logits = self.online_head(r1.detach())
        cls_loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

        # loss = loss.div((loss / cls_loss).detach()) + cls_loss
        return loss, reg, cls_loss, acc

if __name__ == '__main__':
    main()
