#!/usr/bin/env python

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn

from operator import mul
from functools import reduce
import math

'''
QuickGELU and LayerNorm w/ fp16 from official CLIP repo
(https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py)
'''
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
        qk_proj_dim: int, v_proj_dim: int, num_heads: int,
        out_dim: int
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)



    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0); assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1); assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        
        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        #self.attn_probs = aff
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        return out

class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,

        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
        use_mot_token: bool = False,
        num_frames: int = 8,
    ):
        super().__init__()

        self.attn = Attention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)

        self.use_mot_token = use_mot_token
        self.num_frames = num_frames

        if self.use_mot_token:
            self.mot_cls_proj = nn.Linear(in_feature_dim, in_feature_dim)
            self.mot_cls_ln = LayerNorm(in_feature_dim)
            
            self.sem_mot_proj = nn.Linear(in_feature_dim, in_feature_dim)
            self.sem_mot_ln = LayerNorm(in_feature_dim)
            self.sem_mot_attn_layer = Attention(
                                    q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
                                    qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim
                                )

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def _initialize_cls_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.local_prompts.data, -val, val)


    def forward(self, i, x):
        BT, N, C = x.shape
        T = self.num_frames
        B = BT//T
        if self.use_mot_token:
            cls_token = x[:, 0, :].view(B, T, C)            
            img_token = x[:, 1:, :].view(B, T, N-1, C)
        if self.use_mot_token:
            z = []
            for i in range(6):
                sem_t_fea, _ = cls_token.split([T - i - 1, i+ 1], dim=1)
                _, sem_tplus_fea = cls_token.split([i+ 1, T - i - 1], dim=1)
                sem_motion_fea = sem_tplus_fea - sem_t_fea
                z.append(sem_motion_fea)            
            sem_mot_token = torch.cat(z, dim=1)
            
            cls_token_proj = self.mot_cls_proj(cls_token)
            cls_token_norm = self.mot_cls_ln(cls_token_proj)

            sem_mot_token_proj = self.sem_mot_proj(sem_mot_token)
            sem_mot_tokem_norm = self.sem_mot_ln(sem_mot_token_proj)
            sem_mot_token_attn = cls_token_proj + self.sem_mot_attn_layer(cls_token_norm, sem_mot_tokem_norm, sem_mot_tokem_norm)

            sem_mot_token_attn = sem_mot_token_attn.view(BT, 1, C)

            x = torch.cat([x, sem_mot_token_attn], dim=1)

        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)

        if self.use_mot_token:
            x = x[:, :-1, :]

        x = x + self.mlp(self.norm2(x))

        return x


class ImagePatchEmbed2D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x