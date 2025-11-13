from typing import Tuple
import numpy as np
from einops import rearrange
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import mul
from functools import reduce
import math
from clip.VitaCLIP_vision_encoder_utils import QuickGELU, LayerNorm, TransformerEncoderLayer, ImagePatchEmbed2D


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
        self.q_ln = LayerNorm(q_in_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.k_ln = LayerNorm(k_in_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.v_ln = LayerNorm(v_in_dim)
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
        q, k, v = self.q_ln(q), self.k_ln(k), self.v_ln(v)
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H
        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)
        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)
        out = self.out_proj(mix.flatten(-2))

        return out
        
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
                
class CLIPVisionEncoder(nn.Module):

    def __init__(
        self,
        batch_size: int=16,
        input_size: Tuple[int, int] = (224, 224),
        num_frames: int = 8,
        feature_dim: int = 768,
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        embed_dim: int = 512,
        use_mot_token: bool = False,

    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.patch_embed = ImagePatchEmbed2D(img_size=input_size[0], patch_size=patch_size[0], in_chans=3, embed_dim=feature_dim)
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, feature_dim]))
        self.time_embed = nn.Parameter(torch.zeros([8, feature_dim]))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                batch_size=batch_size, in_feature_dim=feature_dim, qkv_dim=feature_dim, num_heads=num_heads,
                mlp_factor=mlp_factor, act=act, use_mot_token=use_mot_token,
                num_frames=num_frames, patch_size=patch_size
            ) for _ in range(num_layers)
        ])

        self.ln_pre = LayerNorm(feature_dim)
        self.ln_post = LayerNorm(feature_dim)
        scale = feature_dim ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(feature_dim, embed_dim))
        
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

    def temporal_encoding(self, x, T, B):
        ## Time Embeddings
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(0):
            time_embed = self.time_embed.unsqueeze(0).transpose(1,2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2).squeeze(0)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
        x = rearrange(x, '(b n) t m -> (b t) n m',b=B,t=T)
        return x

    def forward(self, x: torch.Tensor):
  
        BT, C, H, W = x.size()
        B = BT // self.num_frames
        T = self.num_frames
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.view(1, 1, -1).repeat(x.size(0), 1, 1), x], dim=1)
        x = x + self.pos_embed
        x = self.temporal_encoding(x, T, B)
        x = self.ln_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(i, x)
        cls_x = self.ln_post(x[:, 0, :])
        cls_x = cls_x @ self.proj
        cls_x = rearrange(cls_x, '(b t) e -> b t e', b=B,t=T)
        cls_x = cls_x.mean(dim=1)

        return cls_x
