import os.path as osp

import torch, os
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import models

import torch_geometric.transforms as T
from torch_geometric.nn.models import CorrectAndSmooth

from logger import Logger
import argparse


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.lins = ModuleList([Linear(in_channels, hidden_channels)])
        self.bns = ModuleList([BatchNorm1d(hidden_channels)])

        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        self.lins.append(Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lins in self.lins:
            lins.reset_parameters()
        for bns in self.bns:
            bns.reset_parameters()

    def forward(self, x):
        for lin, bn in zip(self.lins[:-1], self.bns):
            x = bn(lin(x).relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)


def spectral(data, post_fix):
    '''
    generate spectral embeddings, save the results in ./spectral{post_fix}.pt
    '''
    from julia.api import Julia
    jl = Julia(runtime='/home/zhijie/julia-1.8.1/bin/julia', compiled_modules=False)
    # jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./norm_spec.jl")

    print('Setting up spectral embedding', flush=True)
    adj = data.adj_t
    adj = adj.to_scipy(layout='csr')
    result = torch.tensor(Main.main(adj, 128)).float()
    print('Done!', flush=True)

    torch.save(result, f'./spectral{post_fix}.pt')

    return result


def process_adj(data, device):
    adj_t = data.adj_t.to(device)
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t

    return DAD, DA


def train(model, optimizer, x_train, criterion, y_train):
    model.train()
    optimizer.zero_grad()
    out = model(x_train)
    loss = criterion(out, y_train.view(-1))
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=None):
    model.eval()
    out = model(x) if out is None else out
    pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': y[train_idx],
        'y_pred': pred[train_idx]
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y[val_idx],
        'y_pred': pred[val_idx]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[test_idx],
        'y_pred': pred[test_idx]
    })['acc']
    return train_acc, val_acc, test_acc, out


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (MLP-CS)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--use_embed", action="store_true")
    parser.add_argument("--use_cached", action="store_true")
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--label_ratio', default=1., type=float)
    args = parser.parse_args()
    print(args, flush=True)

    if os.path.exists('/data/zhijie/data'):
        args.root = '/data/zhijie/data'
    elif os.path.exists('/home/zhijie/data'):
        args.root = '/home/zhijie/data'
    elif os.path.exists('/workspace/home/zhijie/data'):
        args.root = '/workspace/home/zhijie/data'
    else:
        pass

    dataset = PygNodePropPredDataset('ogbn-products',
                                     root=args.root,
                                     transform=T.ToSparseTensor())
    print(dataset, flush=True)
    evaluator = Evaluator(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    print(data, flush=True)

    device = torch.device("cuda:%d" % args.device if torch.cuda.is_available() else 'cpu')

    # generate and add embeddings
    if args.use_embed:
        if args.use_cached:
            embeddings = torch.load('./spectralproducts.pt', map_location='cpu')
        else:
            embeddings = spectral(data, 'products')
        data.x = torch.cat([data.x, embeddings], dim=-1)

    x, y = data.x.to(device), data.y.to(device)

    # MLP-Wide
    model = MLP(x.size(-1),
                dataset.num_classes,
                hidden_channels=args.hidden_channels,
                num_layers=args.num_layers,
                dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train_idx = split_idx['train'].to(device)
    val_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        print(sum(p.numel() for p in model.parameters()), flush=True)

        train_idx_ = train_idx[torch.randperm(len(train_idx), device=train_idx.device)[:int(len(train_idx) * args.label_ratio)]]
        x_train, y_train = x[train_idx_], y[train_idx_]

        print('', flush=True)
        print(f'Run {run + 1:02d}:', flush=True)
        print('', flush=True)

        best_val_acc = 0
        for epoch in range(1, args.epochs+ 1):  ##
            loss = train(model, optimizer, x_train, criterion, y_train)
            train_acc, val_acc, test_acc, out = test(model, x, evaluator, y, train_idx, val_idx, test_idx)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                y_soft = out.softmax(dim=-1)

            print(
                f'Run: {run + 1:02d}, '
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
                flush=True)

        # DAD, DA = process_adj(data, device)
        #
        # post = CorrectAndSmooth(num_correction_layers=50,
        #                         correction_alpha=1.0,
        #                         num_smoothing_layers=50,
        #                         smoothing_alpha=0.8,
        #                         autoscale=False,
        #                         scale=15.)
        #
        # print('Correct and smooth...', flush=True)
        # y_soft = post.correct(y_soft, y_train, train_idx_, DAD)
        # y_soft = post.smooth(y_soft, y_train, train_idx_, DA)
        # print('Done!', flush=True)

        # train_acc, val_acc, test_acc, _ = test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=y_soft)

        # print(
        #     f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
        #     flush=True)

        result = (train_acc, val_acc, test_acc)
        logger.add_result(run, result)

    logger.print_statistics()


if __name__ == '__main__':
    main()
