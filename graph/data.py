import numpy as np
import torch
import torch.nn as nn
from ogb.nodeproppred import NodePropPredDataset, DglNodePropPredDataset, Evaluator
import torch.nn.functional as F
import gc
from cogdl.data import Graph
from cogdl.utils import to_undirected, spmm_cpu


def get_ogb_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]


def load_dataset_products(root, name, device):

    dataset = NodePropPredDataset(name=name, root=root)
    splitted_idx = dataset.get_idx_split()
    graph, y = dataset[0]
    x = torch.tensor(graph["node_feat"]).float().contiguous() if graph["node_feat"] is not None else None
    y = torch.LongTensor(y.squeeze())
    row, col = graph["edge_index"][0], graph["edge_index"][1]
    row = torch.from_numpy(row)
    col = torch.from_numpy(col)
    edge_index = torch.stack([row, col], dim=0)

    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    assert np.max(train_nid) <= np.min(val_nid)
    assert np.max(val_nid) <= np.min(test_nid)
    evaluator = get_ogb_evaluator(name)

    print(f"# Nodes: {graph.num_nodes}\n"
          f"# Edges: {graph.num_edges}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {graph.num_classes}\n")

    train_nid = torch.LongTensor(train_nid)
    val_nid = torch.LongTensor(val_nid)
    test_nid = torch.LongTensor(test_nid)
    return x, y, edge_index, train_nid, val_nid, test_nid, evaluator

def load_dataset_papers(root, name, device):

    dataset = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = dataset.get_idx_split()
    graph, y = dataset[0]
    x = torch.tensor(graph["node_feat"]).float().contiguous() if graph["node_feat"] is not None else None
    y = torch.LongTensor(y.squeeze())
    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    g, _ = dataset[0]
    #g = None
    n_classes = dataset.num_classes
    labels = labels.squeeze()

    labels = labels.to(torch.long)
    n_classes = max(labels) + 1
    evaluator = get_ogb_evaluator(name)
    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}\n")

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator

def prepare_data(device, args):
    if args.dataset == 'ogbn-papers100M':
        data = load_dataset_papers(args.dataset, device, args)
        g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
        gc.collect()
        feats=[]
        for i in range(args.num_hops+1):
            feats.append(torch.load(f"./papers100m_feat_{i}.pt"))
        in_feats=feats[0].shape[1]

        train_nid = train_nid.to(device)
        val_nid = val_nid.to(device)
        test_nid = test_nid.to(device)
        labels = labels.to(device).to(torch.long)
        return feats, torch.cat([labels[train_nid], labels[val_nid], labels[test_nid]]),int(in_feats), int(n_classes), \
                train_nid, val_nid, test_nid, evaluator, label_emb
    elif args.dataset == 'ogbn-products':
        data = load_dataset_products(args.dataset, device, args)

        graph, train_node_nums, valid_node_nums, test_node_nums, evaluator = data
        if args.dataset == 'ogbn-papers100M':
            graph.edge_index = to_undirected(graph.edge_index)

        # move to device
        feats=[]
        for i in range(args.num_hops+1):
            if args.giant:
                print(f"load feat_{i}_giant.pt")
                feats.append(torch.load(f"./{args.dataset}/feat/{args.dataset}_feat_{i}_giant.pt"))
            else:
                print(f"load feat_{i}.pt")
                feats.append(torch.load(f"./{args.dataset}/feat/{args.dataset}_feat_{i}.pt"))
        in_feats=feats[0].shape[1]

        if args.dataset == 'ogbn-products':
            return graph, feats, graph.y, in_feats, graph.num_classes, \
                   train_node_nums, valid_node_nums, test_node_nums, evaluator
    else:
        raise NotImplementedError
