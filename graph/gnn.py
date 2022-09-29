import argparse, os, time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


import torch.nn as nn

from timm.models.layers import DropPath

import numpy as np
import math

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=False):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Dropout(p=dropout, inplace=True))
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_channels if _ == 0 else hidden_channels, hidden_channels, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers)
        self.head = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.head(self.layers(x))
        return x

    def no_weight_decay(self):
        return {}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes, drop_path):
        super(BasicBlock, self).__init__()
        self.linear1 = nn.Linear(planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.linear2 = nn.Linear(planes, planes, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        out = self.drop_path(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, drop_path=0.):
        super(ResMLP, self).__init__()
        self.first_dropout = nn.Dropout(p=dropout, inplace=True)
        self.linear1 = nn.Linear(in_channels, hidden_channels,  bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.res_layers = self._make_layer(BasicBlock, hidden_channels, num_layers // 2 - 1, drop_path)
        self.head = nn.Linear(hidden_channels, out_channels)

    def _make_layer(self, block, planes, num_blocks, drop_path):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, drop_path))
        return nn.Sequential(*layers)

    def no_weight_decay(self):
        return {}

    def forward(self, x):
        out = F.relu(self.bn1(self.linear1(self.first_dropout(x))))
        out = self.res_layers(out)
        out = self.head(out)
        return out


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--label_ratio', default=1., type=float)
    args = parser.parse_args()
    print(args)

    if os.path.exists('/data/zhijie/data'):
        args.root = '/data/zhijie/data'
    elif os.path.exists('/home/zhijie/data'):
        args.root = '/home/zhijie/data'
    elif os.path.exists('/workspace/home/zhijie/data'):
        args.root = '/workspace/home/zhijie/data'
    else:
        pass

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products',
                                     root=args.root,
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-products')

    if 0:
        model.eval()

        hidden_channels = 2048
        num_layers = 12
        dropout = 0
        our_model = ResMLP(data.x.size(-1), hidden_channels, dataset.num_classes, num_layers, dropout)
        our_model = our_model.to(device)
        our_model.eval()

        i = 0
        time_spent = []
        while i < 100:
            start_time = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    model(data.x, data.adj_t)[split_idx['test']]

            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            if i != 0:
                time_spent.append(time.time() - start_time)
            i += 1
        print('Avg execution time (s): {:.4f}'.format(np.mean(time_spent)))

        batch_size = len(split_idx['test']) // 4
        text_x = data.x[split_idx['test']]
        i = 0
        time_spent = []
        while i < 100:
            start_time = time.time()
            with torch.no_grad():
                for j in range(int(math.ceil(len(split_idx['test']) / float(batch_size)))):
                    with torch.cuda.amp.autocast():
                        our_model(text_x[j * batch_size : min((j+1) * batch_size, len(split_idx['test']))])

            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            if i != 0:
                time_spent.append(time.time() - start_time)
            i += 1
        print('Avg execution time (s): {:.4f}'.format(np.mean(time_spent)))

        i = 0
        time_spent = []
        while i < 100:
            start_time = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    our_model(data.x[split_idx['test'][0:1]])

            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            if i != 0:
                time_spent.append(time.time() - start_time)
            i += 1
        print('Avg execution time (s): {:.4f}'.format(np.mean(time_spent)))
        exit()

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        train_idx_ = train_idx[torch.randperm(len(train_idx), device=train_idx.device)[:int(len(train_idx) * args.label_ratio)]]
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_acc = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx_, optimizer)
            result = test(model, data, split_idx, evaluator)

            train_acc, valid_acc, test_acc = result
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_result = (train_acc, valid_acc, test_acc)

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.add_result(run, best_result)
    logger.print_statistics()


if __name__ == "__main__":
    main()
