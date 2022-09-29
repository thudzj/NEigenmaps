import argparse, scipy, os, math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.utils import get_laplacian, add_self_loops
from torch_geometric.nn.models import CorrectAndSmooth

from torch_sparse.tensor import SparseTensor
from torch_sparse.diag import fill_diag, set_diag, remove_diag
from torch_sparse.cat import cat

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from timm.optim.optim_factory import param_groups_weight_decay

from logger import Logger
from res_mlp import MLP, ResMLP
from utils import LARS, exclude_bias_and_norm, param_groups_lrd, process_adj, adjust_learning_rate

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class NeuralEFCLR(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone = backbone
        in_features = self.backbone.head.in_features
        self.online_head = nn.Linear(in_features, args.num_classes)
        self.backbone.head = nn.Identity()

        if args.proj_dim[0] == 0:
            self.projector = nn.Identity()
        else:
            if len(args.proj_dim) > 1:
                sizes = [in_features,] + args.proj_dim
                layers = []
                for i in range(len(sizes) - 2):
                    layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                    layers.append(nn.BatchNorm1d(sizes[i+1]))
                    layers.append(nn.ReLU(inplace=True))

                layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))

                self.projector = nn.Sequential(*layers)
            elif len(args.proj_dim) == 1:
                self.projector = nn.Linear(in_features, args.proj_dim[0], bias=True)

        print(self.projector)

    def forward(self, x, y=None, K=None, mask=None, make_prediction=False):
        r = self.backbone(x)
        if make_prediction:
            return self.online_head(r.detach())

        psi = self.projector(r)
        norm_ = psi.norm(dim=0).clamp(min=1e-6)
        psi = psi.div(norm_) * math.sqrt(2 * self.args.t)
        psi1, psi2 = psi[:K.shape[0]], psi[K.shape[0]:]

        psi_K_psi_diag = (psi1.T @ (K @ psi2)).diag().view(-1, 1)
        if self.args.no_stop_grad:
            psi2_d_K_psi1 = psi2.T @ (K.T @ psi1)
            psi1_d_K_psi2 = psi1.T @ (K @ psi2)
        else:
            psi2_d_K_psi1 = psi2.detach().T @ (K.T @ psi1)
            psi1_d_K_psi2 = psi1.detach().T @ (K @ psi2)

        loss = - psi_K_psi_diag.sum() * 2
        reg = ((psi2_d_K_psi1) ** 2).triu(1).sum() \
            + ((psi1_d_K_psi2) ** 2).triu(1).sum()
        loss /= psi_K_psi_diag.numel()
        reg /= psi_K_psi_diag.numel()

        logits = self.online_head(r.detach())
        cls_loss = (F.cross_entropy(logits, y.squeeze(1), reduction='none') * mask).sum() / mask.sum()
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), y.squeeze(1)) * mask) / mask.sum()

        return loss, reg, cls_loss, acc

    def reset_parameters(self):
        if self.args.model == 'mlp' or self.args.model == 'res_mlp':
            for layer in self.backbone.modules():
                if isinstance(layer, (nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                    layer.reset_parameters()
        else:
            self.backbone.init_weights()

        for layer in self.projector.modules():
            if isinstance(layer, (nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                layer.reset_parameters()
        self.online_head.reset_parameters()

def batchify1v(x, y, K, train_idx, batch_size):
    num_nodes = K.sparse_sizes()[0]
    permutation_i = torch.randperm(num_nodes, device=x.device)
    mask_i_all = torch.isin(permutation_i, train_idx).float()
    permutation_j = torch.randperm(num_nodes, device=x.device)
    mask_j_all = torch.isin(permutation_j, train_idx).float()
    for i in range(num_nodes // batch_size):
        idx_i = permutation_i[i * batch_size : (i+1) * batch_size]
        mask_i = mask_i_all[i * batch_size : (i+1) * batch_size]
        x_i, y_i = x[idx_i], y[idx_i]
        K_i = K[idx_i, :]
        for j in range(num_nodes // batch_size):
            idx_j = permutation_j[j * batch_size : (j+1) * batch_size]
            mask_j = mask_j_all[j * batch_size : (j+1) * batch_size]
            x_j, y_j = x[idx_j], y[idx_j]
            K_ij = K_i[:, idx_j]

            row, col, _ = K_ij.coo()
            node_idx1, node_idx2 = row.unique(sorted=True), col.unique(sorted=True)
            K_ = K_ij[node_idx1, :][:, node_idx2]
            x_ = torch.cat([x_i[node_idx1], x_j[node_idx2]])
            y_ = torch.cat([y_i[node_idx1], y_j[node_idx2]])
            mask = torch.cat([mask_i[node_idx1], mask_j[node_idx2]])
            yield x_, y_, K_.to_dense(), mask

def batchify2(x, y, K, train_idx, batch_size):
    num_nodes = K.sparse_sizes()[0]
    permutation_i = torch.randperm(num_nodes, device=x.device)
    mask_i_all = torch.isin(permutation_i, train_idx).float()
    permutation_j = torch.randperm(num_nodes, device=x.device)
    mask_j_all = torch.isin(permutation_j, train_idx).float()
    for i in range(num_nodes // batch_size):
        idx_i = permutation_i[i * batch_size : (i+1) * batch_size]
        mask_i = mask_i_all[i * batch_size : (i+1) * batch_size]
        x_i, y_i = x[idx_i], y[idx_i]
        K_i = K[idx_i, :]
        for j in range(num_nodes // batch_size):
            idx_j = permutation_j[j * batch_size : (j+1) * batch_size]
            mask_j = mask_j_all[j * batch_size : (j+1) * batch_size]
            x_j, y_j = x[idx_j], y[idx_j]
            K_j = K[idx_j, :]
            K_ij = K_i @ K_j.t()

            row, col, _ = K_ij.coo()
            node_idx1, node_idx2 = row.unique(sorted=True), col.unique(sorted=True)
            K_ = K_ij[node_idx1, :][:, node_idx2]
            x_ = torch.cat([x_i[node_idx1], x_j[node_idx2]])
            y_ = torch.cat([y_i[node_idx1], y_j[node_idx2]])
            mask = torch.cat([mask_i[node_idx1], mask_j[node_idx2]])
            yield x_, y_, K_.to_dense(), mask

def batchify(x, y, K, indices, batch_size, drop_last=False):
    permutation = torch.randperm(len(indices), device=x.device)
    if drop_last:
        for i in range(len(indices) // batch_size):
            idx = indices[permutation[i * batch_size : (i+1) * batch_size]]
            yield x[idx], y[idx]
    else:
        for i in range(int(math.ceil(len(indices) / float(batch_size)))):
            idx = indices[permutation[i * batch_size : min((i+1) * batch_size, len(indices))]]
            yield x[idx], y[idx]


def finetune(model, run, args, device, train_batches, data, evaluator, train_idx, valid_idx, test_idx):
    if args.model == 'mlp':
        ft_model = MLP(args.feature_dim, args.hidden_channels, args.num_classes,
            args.num_layers, args.dropout, args.use_bn)
    elif args.model == 'res_mlp':
        ft_model = ResMLP(args.feature_dim, args.hidden_channels, args.num_classes, args.num_layers, args.dropout)
    else:
        raise NotImplementedError

    if not args.ft_from_sractch:
        missing_keys, unexpected_keys = ft_model.load_state_dict(model.backbone.state_dict(), strict=False)
        assert missing_keys == ['head.weight', 'head.bias']
        assert unexpected_keys == []

    if args.ft_mode == 'freeze':
        ft_model.requires_grad_(False)
        ft_model.head.requires_grad_(True)
        param_groups = param_groups_weight_decay(ft_model.head, args.ft_weight_decay)
    else:
        param_groups = param_groups_lrd(ft_model, args.ft_weight_decay, layer_decay=args.ft_layer_decay)
    optimizer = torch.optim.SGD(param_groups, args.ft_lr, momentum=0.9)

    ft_model = ft_model.to(device)
    scheduler = CosineLRScheduler(optimizer,
            t_initial=args.ft_epochs,
            lr_min=0,
            warmup_lr_init=1e-6,
            warmup_t=10)

    if args.ft_mixup > 0:
        mixup_fn = Mixup(
                mixup_alpha=args.ft_mixup,
                prob=args.ft_mixup_prob, mode=args.ft_mixup_mode,
                label_smoothing=args.ft_smoothing, num_classes=args.num_classes)
        loss_fn = SoftTargetCrossEntropy()
    elif args.ft_smoothing > 0:
        mixup_fn = None
        loss_fn = LabelSmoothingCrossEntropy(smoothing=args.ft_smoothing)
    else:
        mixup_fn = None
        loss_fn = torch.nn.CrossEntropyLoss()

    best_valid_acc = 0
    for epoch in range(args.ft_epochs):
        if args.ft_mode == 'freeze':
            ft_model.eval()
            ft_model.head.train()
        else:
            ft_model.train()
        scheduler.step(epoch)
        losses, num_data, ite = 0, 0, 0
        for batch in train_batches():
            x_batch, y_batch = batch
            y_batch = y_batch.squeeze(1)
            if mixup_fn is not None:
                x_batch, y_batch = mixup_fn(x_batch, y_batch)
            loss = loss_fn(ft_model(x_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.detach() * x_batch.shape[0]
            num_data += x_batch.shape[0]
            ite += 1

        loss = losses / num_data

        if epoch % 1 == 0 or epoch == 99:
            train_acc, valid_acc, test_acc, out = test(ft_model, data.x, evaluator, data.y, train_idx, valid_idx, test_idx)
            if valid_acc > best_valid_acc:
                y_soft_o = out.softmax(dim=-1)
                best_valid_acc = valid_acc
            # print(f'Run: {run + 1:02d}, '
            #       f'fine-tuning, '
            #       f'Epoch: {epoch:02d}, '
            #       f'Loss: {loss:.4f}, '
            #       f'Train: {100 * train_acc:.2f}%, '
            #       f'Valid: {100 * valid_acc:.2f}%, '
            #       f'Test: {100 * test_acc:.2f}% '
            #       )
    return ft_model, loss, y_soft_o

@torch.no_grad()
def test(model, x, evaluator, y, train_idx, val_idx, test_idx, out=None, batch_size=10000):
    model.eval()
    model_fn = model if not isinstance(model, NeuralEFCLR) else partial(model, make_prediction=True)

    if out is None:
        out = []
        for i in range(int(math.ceil(len(x) / float(batch_size)))):
            out.append(model_fn(x[i * batch_size : min((i+1) * batch_size, len(x))]))
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
    parser = argparse.ArgumentParser(description='OGBN-Products')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1000)
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--output_dir', default=None, help='path where to save, empty for no saving')

    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--dataset', type=str, default='products')

    # for specifying MLP
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_bn', default=False, action='store_true')

    # opt configs
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=0.3)
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--epochs', type=int, default=100)

    # for specifying kernel
    parser.add_argument('--use_K_sqr',  default=False, action='store_true')
    parser.add_argument('--K_normalize',  default=False, action='store_true')

    # for neuralefclr
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--proj_dim', default=[2048, 2048], type=int, nargs='+')
    parser.add_argument('--t', default=10, type=float)
    parser.add_argument('--no_stop_grad', default=False, action='store_true')

    # for ft
    parser.add_argument('--ft_label_ratio', default=1., type=float)
    parser.add_argument('--ft_from_sractch', default=False, action='store_true')
    parser.add_argument('--ft_only', default=False, action='store_true')
    parser.add_argument('--ft_mode', default='freeze', type=str,
                        choices=('finetune', 'freeze'),
                        help='finetune or freeze resnet weights')
    parser.add_argument('--ft_lr', default=0.1, type=float, metavar='LR')
    parser.add_argument('--ft_epochs', default=100, type=int)
    parser.add_argument('--ft_layer_decay', default=0.75, type=float)
    parser.add_argument('--ft_weight_decay', type=float, default=1e-6)
    parser.add_argument('--ft_batch_size', type=int, default=256)
    parser.add_argument('--ft_smoothing', type=float, default=0)
    parser.add_argument('--ft_dropout', default=0., type=float)
    parser.add_argument('--ft_mixup', type=float, default=0)
    parser.add_argument('--ft_mixup_prob', type=float, default=1.0)
    parser.add_argument('--ft_mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    args = parser.parse_args()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256
    if args.min_lr is None:
        args.min_lr = args.lr * 0.001
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if os.path.exists('/data/zhijie/data'):
        args.root = '/data/zhijie/data'
    elif os.path.exists('/home/zhijie/data'):
        args.root = '/home/zhijie/data'
    else:
        pass

    if args.dataset == 'products':
        dataset = PygNodePropPredDataset(name='ogbn-products', root=args.root,
                                         transform=T.ToSparseTensor())
    elif args.dataset == 'arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.root,
                                         transform=T.ToSparseTensor())
    else:
        raise NotImplementedError

    data = dataset[0]
    num_nodes = data.num_nodes
    args.feature_dim = data.x.size(-1)
    args.num_classes = dataset.num_classes

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    if args.model == 'mlp':
        model = MLP(args.feature_dim, args.hidden_channels, args.num_classes,
            args.num_layers, args.dropout, args.use_bn)
    elif args.model == 'res_mlp':
        model = ResMLP(args.feature_dim, args.hidden_channels, args.num_classes, args.num_layers, args.dropout)
    else:
        raise NotImplementedError

    print(model)
    model = NeuralEFCLR(args, model).to(device)
    print("# params: {}".format(sum([p.numel() for p in model.backbone.parameters() if p.requires_grad])))

    if args.dataset == 'arxiv':
        data.adj_t = data.adj_t.to_symmetric()
    original_adj_t = data.adj_t

    if args.use_K_sqr:
        deg = data.adj_t.sum(dim=1).to(torch.float)#.fill_(1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * data.adj_t * deg_inv_sqrt.view(1, -1)
        adj_t = fill_diag(adj_t, 1) # K = fill_diag(adj_t, args.a - 1)
    else:
        if args.K_normalize:
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)#.fill_(1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        else:
            print(data.adj_t.get_diag().sum())
            adj_t = data.adj_t
            # if not adj_t.has_value():
            #     adj_t = adj_t.fill_value(0.)
            adj_t = remove_diag(adj_t)
            deg = adj_t.sum(dim=1).to(torch.float)#.fill_(1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            # adj_t = adj_t.set_diag()

    K = adj_t
    del data.adj_t
    data = data.to(device)
    K = K.to(device)

    print(num_nodes, K.sparse_sizes(), K.is_symmetric(), K.get_diag().sum(), K)
    print(train_idx)
    print(train_idx.shape, train_idx.unique().shape, train_idx.max(), train_idx.min())
    evaluator = Evaluator(name='ogbn-products')

    if not args.ft_only:
        model.reset_parameters()

        optimizer = LARS(model.parameters(),
                        lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=exclude_bias_and_norm,
                        lars_adaptation_filter=exclude_bias_and_norm)
        scaler = torch.cuda.amp.GradScaler()

        batchify_fn = batchify2 if args.use_K_sqr else batchify1v
        train_batches_neuralef = partial(batchify_fn, data.x, data.y, K, train_idx, args.batch_size)

        for epoch in range(args.epochs):
            lr = adjust_learning_rate(optimizer, epoch, args.lr, args.min_lr, args.epochs, args.warmup_epochs)
            model.train()
            losses, regs, cls_losses, accs, num_data, ite = 0, 0, 0, 0, 0, 0
            avg_h, avg_w = 0, 0
            for batch in train_batches_neuralef():

                x_batch, y_batch, K_batch, mask_batch = batch
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss, reg, cls_loss, acc = model.forward(x_batch, y_batch, K_batch, mask_batch)
                scaler.scale(loss + reg * args.alpha + cls_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses += loss.detach()
                regs += reg.detach()
                cls_losses += cls_loss.detach()
                accs += acc.detach()
                num_data += x_batch.shape[0]
                avg_h += K_batch.shape[0]
                avg_w += K_batch.shape[1]
                ite += 1

                if ite % args.log_steps == 0:
                    print(f'Epoch: {epoch:02d}, '
                          f'Ite: {ite}, '
                          f'LR: {lr:.4f}, '
                          f'Loss: {losses / ite:.4f}, '
                          f'Reg: {regs / ite:.4f}, '
                          f'size: ({avg_h / float(ite):.1f}, {avg_w / float(ite):.1f}), '
                          f'CLS_Loss: {cls_losses / ite:.4f}, '
                          f'Online: {100 * accs / ite:.2f}%'
                          )

            # loss, reg, cls_loss, online_acc = \
            #     losses / num_data, regs / num_data, cls_losses / num_data, accs / num_data
            # avg_h /= float(ite); avg_w /= float(ite);
            train_acc, valid_acc, test_acc, _ = test(model, data.x, evaluator, data.y, train_idx, valid_idx, test_idx)
            print(f'Test -- Epoch: {epoch:02d}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%')



            if (args.save_steps is not None and epoch % args.save_steps == 0) or epoch == args.epochs - 1:
                if args.output_dir:
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    to_save = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': scaler.state_dict(),
                        'args': args,
                    }
                    torch.save(to_save, os.path.join(args.output_dir, 'checkpoint{}_e{}.pth'.format(0, epoch)))

    logger = Logger(args.runs, args)
    logger1 = Logger(args.runs, args)
    state_dict = torch.load(os.path.join(args.output_dir, 'checkpoint{}_e{}.pth'.format(0, args.epochs - 1)), map_location='cpu')['model']
    model.load_state_dict(state_dict)
    DAD, DA = process_adj(original_adj_t, device)

    for run in range(args.runs):
        ft_idx = train_idx[torch.randperm(len(train_idx), device=train_idx.device)[:int(len(train_idx) * args.ft_label_ratio)]]
        ft_batches = partial(batchify, data.x, data.y, K, ft_idx, args.ft_batch_size, drop_last=True)
        ft_model, ft_loss, y_soft_o = finetune(model, run, args, device, ft_batches, data, evaluator, train_idx, valid_idx, test_idx)
        train_acc, valid_acc, test_acc, _ = test(ft_model, data.x, evaluator, data.y, train_idx, valid_idx, test_idx, out=y_soft_o)
        logger1.add_result(run, (train_acc, valid_acc, test_acc))
        print(f'**Post-finetuning results** Run: {run + 1:02d}, '
              f'loss: {ft_loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}%, '
              f'Test: {100 * test_acc:.2f}%')

        # best_test_acc = 0
        # best_config = None
        # 40, 1.1, 40, 0.8, 5
        ##################################################
        # for num_correction_layers in [25, 30, 35, 40, 45, 50]:
        #     for correction_alpha in [0.8, 0.9, 1., 1.1, 1.2]:
        #         for num_smoothing_layers in [25, 30, 35, 40, 45, 50]:
        #             for smoothing_alpha in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        #                 for scale in [1, 3, 5, 7, 10]:
        post = CorrectAndSmooth(num_correction_layers=40,
                                correction_alpha=1.1,
                                num_smoothing_layers=25,
                                smoothing_alpha=0.75,
                                autoscale=False,
                                scale=3
                                )

        # print('Correct and smooth...', flush=True)
        # print(num_correction_layers, correction_alpha, num_smoothing_layers, smoothing_alpha, scale)
        y_soft = post.correct(y_soft_o, data.y[ft_idx], ft_idx, DAD)
        y_soft = post.smooth(y_soft, data.y[ft_idx], ft_idx, DA) # DAD
        # print('Done!', flush=True)
        train_acc, valid_acc, test_acc, _ = test(ft_model, data.x, evaluator, data.y, train_idx, valid_idx, test_idx, out=y_soft)
        # if test_acc > best_test_acc:
        #     best_test_acc = test_acc
        #     best_config = (num_correction_layers, correction_alpha, num_smoothing_layers, smoothing_alpha, scale)
        print(
            f'**C-S results** Train: {train_acc:.4f}, Val: {valid_acc:.4f}, Test: {test_acc:.4f}',
            flush=True)
        # print("best test", best_test_acc, "best config", best_config)
        logger.add_result(run, (train_acc, valid_acc, test_acc))
    logger1.print_statistics()
    logger.print_statistics()



if __name__ == "__main__":
    main()
