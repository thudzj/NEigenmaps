import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

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
