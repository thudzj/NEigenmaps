# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class CustomTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, chunk_len, feature_dim, **kwargs):
        assert feature_dim % chunk_len == 0
        model_kwargs = dict(img_size=(feature_dim, 1), patch_size=(chunk_len, 1), in_chans=1, global_pool='avg', **kwargs)
        super(CustomTransformer, self).__init__(**model_kwargs)
        assert self.patch_embed.num_patches == feature_dim // chunk_len, (self.patch_embed.num_patches, feature_dim, chunk_len)

        self.chunk_len = chunk_len
        del self.patch_embed
        self.patch_embed = nn.Linear(chunk_len, self.embed_dim)

    def forward_features(self, x):
        x = self.patch_embed(x.view(x.shape[0], -1, self.chunk_len))
        x = self._pos_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

def custom_transformer_toy(**kwargs):
    model = CustomTransformer(
        embed_dim=128, depth=6, num_heads=2, mlp_ratio=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def custom_transformer_tiny(**kwargs):
    model = CustomTransformer(
        embed_dim=192, depth=12, num_heads=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def custom_transformer_small(**kwargs):
    model = CustomTransformer(
        embed_dim=384, depth=12, num_heads=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def custom_transformer_base(**kwargs):
    model = CustomTransformer(
        embed_dim=768, depth=12, num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def custom_transformer_large(**kwargs):
    model = CustomTransformer(
        embed_dim=1024, depth=24, num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def custom_transformer_huge(**kwargs):
    model = CustomTransformer(
        embed_dim=1280, depth=32, num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
