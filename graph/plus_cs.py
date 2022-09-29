import os.path as osp

import torch, os, math
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


def process_adj(data, device):
    adj_t = data.adj_t.to(device)
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t

    return DAD, DA


def train(model, optimizer, x_train, criterion, y_train, batch_size=200000):
    model.train()
    permutation = torch.randperm(len(x_train))
    loss_ = 0
    for i in range(int(math.ceil(len(x_train) / float(batch_size)))):
        idx = permutation[i * batch_size : min((i+1) * batch_size, len(x_train))]
        optimizer.zero_grad()
        out = model(x_train[idx].to(y_train.device))
        loss = criterion(out, y_train[idx].view(-1))
        loss.backward()
        optimizer.step()
        loss_ += loss.item() * len(idx)
    return float(loss_) / len(x_train)


@torch.no_grad()
def test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=None, batch_size=500000):
    model.eval()
    if out is None:
        out = []
        for i in range(int(math.ceil(len(x) / float(batch_size)))):
            out.append(model(x[i * batch_size : min((i+1) * batch_size, len(x))].to(y.device)))
        out = torch.cat(out)
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
    parser.add_argument("--embed_path", type=str, default="/home/zhijie/data/products_features.pt")
    args = parser.parse_args()
    print(args, flush=True)

    if os.path.exists('/data/zhijie/data'):
        args.root = '/data/zhijie/data'
    elif os.path.exists('/home/zhijie/data'):
        args.root = '/home/zhijie/data'
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

    embeddings = torch.load(args.embed_path, map_location='cpu')
    data.x = torch.cat([data.x, embeddings], dim=-1)

    x, y = data.x, data.y.to(device)

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
    x_train, y_train = x[train_idx], y[train_idx]
    print(x_train.shape, x.shape)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        print(sum(p.numel() for p in model.parameters()), flush=True)

        print('', flush=True)
        print(f'Run {run + 1:02d}:', flush=True)
        print('', flush=True)

        best_val_acc = 0
        for epoch in range(1, args.epochs+ 1):  ##
            loss = train(model, optimizer, x_train, criterion, y_train)
            train_acc, val_acc, test_acc, out = test(model, x, evaluator, y, train_idx, val_idx, test_idx)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                y_soft_o = out.softmax(dim=-1)

            print(
                f'Run: {run + 1:02d}, '
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
                flush=True)

        DAD, DA = process_adj(data, device)

        # best_test_acc = 0
        # best_config = None
        # # 40, 1.1, 40, 0.8, 5
        # ##################################################
        # for num_correction_layers in [30, 40, 50, 60, 70]:
        #     for correction_alpha in [0.8, 0.9, 1., 1.1, 1.2]:
        #         for num_smoothing_layers in [30, 40, 50, 60, 70]:
        #             for smoothing_alpha in [0.8, 0.9, 1., 1.1, 1.2]:
        #                 for scale in [5, 10, 15, 20, 25]:
        #                     post = CorrectAndSmooth(num_correction_layers=num_correction_layers,
        #                                             correction_alpha=correction_alpha,
        #                                             num_smoothing_layers=num_smoothing_layers,
        #                                             smoothing_alpha=smoothing_alpha, # 0.9
        #                                             autoscale=False,
        #                                             scale=scale) # 0.8, 10; 0.9, 20
        #
        #                     # print('Correct and smooth...', flush=True)
        #                     print(num_correction_layers, correction_alpha, num_smoothing_layers, smoothing_alpha, scale)
        #                     y_soft = post.correct(y_soft_o, y_train, train_idx, DAD)
        #                     y_soft = post.smooth(y_soft, y_train, train_idx, DA) # DAD
        #                     print('Done!', flush=True)
        #                     train_acc, val_acc, test_acc, _ = test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=y_soft)
        #                     if test_acc > best_test_acc:
        #                         best_test_acc = test_acc
        #                         best_config = (num_correction_layers, correction_alpha, num_smoothing_layers, smoothing_alpha, scale)
        #                     print(
        #                         f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
        #                         flush=True)
        #                     print("best test", best_test_acc, "best config", best_config)
        #
        # ##################################################
        # print(2)
        post = CorrectAndSmooth(num_correction_layers=40,
                                correction_alpha=1.1,
                                num_smoothing_layers=40,
                                smoothing_alpha=0.8, # 0.9
                                autoscale=False,
                                scale=5.) # 0.8, 10; 0.9, 20

        print('Correct and smooth...', flush=True)
        y_soft = post.correct(y_soft_o, y_train, train_idx, DAD)
        y_soft = post.smooth(y_soft, y_train, train_idx, DA) # DAD
        print('Done!', flush=True)
        train_acc, val_acc, test_acc, _ = test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=y_soft)
        print(
            f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
            flush=True)

        # ##################################################
        # print(3)
        # post = CorrectAndSmooth(num_correction_layers=50,
        #                         correction_alpha=1.0,
        #                         num_smoothing_layers=50,
        #                         smoothing_alpha=0.8, # 0.9
        #                         autoscale=False,
        #                         scale=10.) # 0.8, 10; 0.9, 20
        #
        # print('Correct and smooth...', flush=True)
        # y_soft = post.correct(y_soft_o, y_train, train_idx, DAD)
        # y_soft = post.smooth(y_soft, y_train, train_idx, DA) # DAD
        # print('Done!', flush=True)
        # train_acc, val_acc, test_acc, _ = test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=y_soft)
        # print(
        #     f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
        #     flush=True)
        #
        # ##################################################
        # print(4)
        # post = CorrectAndSmooth(num_correction_layers=50,
        #                         correction_alpha=1.0,
        #                         num_smoothing_layers=50,
        #                         smoothing_alpha=0.8, # 0.9
        #                         autoscale=False,
        #                         scale=20.) # 0.8, 10; 0.9, 20
        #
        # print('Correct and smooth...', flush=True)
        # y_soft = post.correct(y_soft_o, y_train, train_idx, DAD)
        # y_soft = post.smooth(y_soft, y_train, train_idx, DA) # DAD
        # print('Done!', flush=True)
        # train_acc, val_acc, test_acc, _ = test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=y_soft)
        # print(
        #     f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
        #     flush=True)
        #
        # ##################################################
        # print(5)
        # post = CorrectAndSmooth(num_correction_layers=50,
        #                         correction_alpha=1.0,
        #                         num_smoothing_layers=50,
        #                         smoothing_alpha=0.9, # 0.9
        #                         autoscale=False,
        #                         scale=20.) # 0.8, 10; 0.9, 20
        #
        # print('Correct and smooth...', flush=True)
        # y_soft = post.correct(y_soft_o, y_train, train_idx, DAD)
        # y_soft = post.smooth(y_soft, y_train, train_idx, DA) # DAD
        # print('Done!', flush=True)
        # train_acc, val_acc, test_acc, _ = test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=y_soft)
        # print(
        #     f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}',
        #     flush=True)
        #
        # ##################################################
        # print(6)
        # post = CorrectAndSmooth(num_correction_layers=50,
        #                         correction_alpha=1.0,
        #                         num_smoothing_layers=50,
        #                         smoothing_alpha=0.9, # 0.9
        #                         autoscale=False,
        #                         scale=10.) # 0.8, 10; 0.9, 20
        #
        # print('Correct and smooth...', flush=True)
        # y_soft = post.correct(y_soft_o, y_train, train_idx, DAD)
        # y_soft = post.smooth(y_soft, y_train, train_idx, DA) # DAD
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
