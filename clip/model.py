# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu
import copy

from collections import OrderedDict
from typing import Tuple, Union
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip.VitaCLIP_vision_encoder import CLIPVisionEncoder
from clip.VitaCLIP_text_encoder import CLIPTextEncoder, TextPromptLearner, ClsPrompt



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

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

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head,dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)] 
        print('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers
        
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,dropout = None,joint=False, emb_dropout = 0.):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        self.joint = joint
        if joint:
            print('=====using joint space-time====')
            self.time_embedding = nn.Parameter(scale * torch.randn(T, width))
        if emb_dropout > 0:
            print('emb_dropout:{}'.format(emb_dropout))

        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, dropout=dropout)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        if self.joint:
            B = x.shape[0] // self.T
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=self.T)
            x = x + self.time_embedding.to(x.dtype)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=self.T)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x



class CLIP(nn.Module):
    def __init__(self,
                 backbone_path: str,
                 embed_dim: int,
                 feature_dim: int,
                 batch_size: int,
                 # vision
                 num_cls: int,
                 image_resolution: int,
                 num_frames: int,
                 mlp_factor: float,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_heads: int,
                 vision_patch_size: int,
                 # text
                 use_text_prompt_learning: bool = False,
                 use_mot_token: bool = False,
                 text_context_length: int = 77,
                 text_vocab_size: int = 49408,
                 text_transformer_width: int = 512,
                 text_transformer_heads: int = 8,
                 text_transformer_layers: int = 12,
                 text_num_prompts: int = 8,
                 text_prompt_pos: str = 'end',
                 text_prompt_init: str = '',
                 text_prompt_CSC: bool = False,
                 text_prompt_classes_path: str = '',
                 
 

                 ):
        super().__init__()

        self.visual = CLIPVisionEncoder(
            batch_size=batch_size,
            num_cls=num_cls,
            # data shape
            input_size=image_resolution,
            num_frames=num_frames,
            # model def
            feature_dim=feature_dim,
            patch_size=vision_patch_size,
            num_heads=vision_heads,
            num_layers=vision_layers,
            mlp_factor=mlp_factor,
            embed_dim=embed_dim,
            use_mot_token=use_mot_token,

        )
        self.mot_token_ln = LayerNorm(embed_dim)
        self.mot_token_proj = nn.Linear(embed_dim, num_cls)
     
        self.use_text_prompt_learning = use_text_prompt_learning
        if self.use_text_prompt_learning:
            self.textual = CLIPTextEncoder(
                embed_dim=embed_dim,
                context_length=text_context_length,
                vocab_size=text_vocab_size,
                transformer_width=text_transformer_width,
                transformer_heads=text_transformer_heads,
                transformer_layers=text_transformer_layers,
            )

        if self.use_text_prompt_learning:
            with open(text_prompt_classes_path, 'r') as f:
                classes = f.read().strip().split('\n')
            
            self.prompt_learner = TextPromptLearner(
                            classnames=classes,
                            batch_size=batch_size,
                            text_model=self.textual,
                            num_prompts=text_num_prompts,
                            prompts_init=text_prompt_init,
                            CSC=text_prompt_CSC,
                            ctx_pos=text_prompt_pos
                            )
            self.learnable_tokenized_prompts = self.prompt_learner.learnable_tokenized_prompts
            self.cls_prompt = ClsPrompt(
                            classnames=classes,
                            text_model=self.textual,
                            )
        else:
            self.text_features = torch.load('./clip/hmdb_text_features.pth', map_location='cpu')
                            

   
                      
        
        self.mlp1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(embed_dim, 4 * embed_dim)),
            ('act', QuickGELU()),
            ('fc2', nn.Linear(4 * embed_dim, embed_dim)),
        ]))    
        self.mlp2 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(embed_dim, 4 * embed_dim)),
            ('act', QuickGELU()),
            ('fc2', nn.Linear(4 * embed_dim, embed_dim)),
        ]))   


        
        if backbone_path:
            ckpt = torch.load(backbone_path)
            self.load_state_dict(ckpt, strict=False)
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.vision_layers = vision_layers
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self._freeze_visual_except_prompts_time_embed()
        self._freeze_textual()

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.mlp1, self.mlp2):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _freeze_visual_except_prompts_time_embed(self):
        for name, param in self.visual.named_parameters():
                if 'time_embed' in name or 'mot' in name or 'MP' in name or 'tg' in name:
                    pass
                else:
                    param.requires_grad = False
    
    def _freeze_textual(self):
        for name, param in self.textual.named_parameters():
                if 'MTP' in name or 'mot' in name :
                    pass
                else:
                    param.requires_grad = False

    @property
    def dtype(self):
        return self.mot_token_proj.weight.dtype

    def forward(self, image):
        
        if self.use_text_prompt_learning:
            
            image_features = self.visual(image)
            B, D = image_features.shape
            mot_cls_token = self.mot_token_proj(self.mot_token_ln(image_features)) #(B, 101)
            cls_prompts, cls_tokenized_prompts = self.cls_prompt()
            cls_text_features = self.textual(cls_prompts, cls_tokenized_prompts)
            learnable_prompts = self.prompt_learner(cls_text_features)
            learnable_tokenized_prompts = self.learnable_tokenized_prompts          
            text_features = self.textual(learnable_prompts, learnable_tokenized_prompts)
        else:
            # vision side
            image_features = self.visual(image)
            B, D = image_features.shape
            # text side
            text_features = self.text_features.to(image_features.device)

        visual_logits = mot_cls_token.detach().unsqueeze(dim=1)
        image_features_ = (image_features.unsqueeze(dim=1))
        text_features_ = (text_features.unsqueeze(0).expand(B, -1, -1))

        text_features = self.mlp1(text_features + visual_logits.transpose(1, 2) @ image_features_)
        image_features = self.mlp2(image_features + (visual_logits @ text_features_).squeeze(dim=1))
        
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True))
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True))

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        logits = torch.einsum("bd,bkd->bk", image_features, logit_scale * text_features)

        #return logits
        return logits, mot_cls_token

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, tsm=False,T=8,dropout=0., joint=False,emb_dropout=0.,pretrain=True):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  tsm=tsm,T=T,joint=joint,
        dropout=dropout, emb_dropout=emb_dropout
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    if tsm:
        for k in list(state_dict.keys()):
            if k.find("conv1")>-1 and k.find("layer")>-1: 
                n_k = k.split('conv1.')[0]+'conv1.net.'+k.split('conv1.')[1]
                state_dict[n_k] = state_dict.pop(k)
            if k.find("resblocks")>-1 and k.find("visual")>-1: 
                tmp = ''
                for i, t_ in enumerate(k.split('resblocks.')[1].split('.')):
                    if i>=1:
                        tmp += '.' + t_ 
                
                n_k = k.split('resblocks.')[0]+'resblocks.' + k.split('resblocks.')[1].split('.')[0]+'.net'+ tmp
#                 print(n_k)
                state_dict[n_k] = state_dict.pop(k)

    convert_weights(model)
    if pretrain:
        print('loading clip pretrained model!')
        if joint:  #or emb_dropout>0 or dropout>0
            model.load_state_dict(state_dict,strict=False)
        else:
            model.load_state_dict(state_dict)
    else:
        print('not using full clip pretrained model, only visual!')
        
        for k in list(state_dict.keys()):
            if not k.find("visual")>-1: 
                state_dict.pop(k)

        model.load_state_dict(state_dict,strict=False)

    return model.eval()
