import torch
import torch.nn as nn

import copy
from collections import OrderedDict
from typing import Union, List
from pkg_resources import packaging
from clip.prompt import VideoSpecificPrompt
from clip.VitaCLIP_text_encoder_utils import SimpleTokenizer as _Tokenizer

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:

    if isinstance(texts, str):
        texts = [texts]
    
    _tokenizer = _Tokenizer()

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
    


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
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
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        #self.MTP = MotionTextPrompt(embed_dim=512)

    def forward(self, x: torch.Tensor, maple_prompts=None):
        if maple_prompts:
            num_prompts = 8
            for i, blk in enumerate(self.resblocks):
                if i == 0:
                    prefix = x[:1, :, :]
                    suffix = x[1 + num_prompts:, :, :]
                    textual_context = x[1: 1 + num_prompts, :, :] #(8, 101, 512)
                    motion_features = maple_prompts[i] #(B, 512)
                    textual_context = textual_context + self.MTP(textual_context, motion_features.unsqueeze(0).repeat(num_prompts, 1, 1)) #(101, 512)
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                    x = blk(x)
                else:
                    x = blk(x)
               
        else:
            for blk in self.resblocks:
                x = blk(x)
        return x                 


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        context_length: int = 77,
        vocab_size: int = 49408,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,

    ):
        super().__init__()
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, prompts, tokenized_prompts, maple_prompts=None):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        if maple_prompts:
            x = self.transformer(x, maple_prompts)
        else:
            x = self.transformer(x)
            
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class TextPromptLearner(nn.Module):
    def __init__(self, classnames, batch_size, text_model, num_prompts, prompts_init='', CSC=False, ctx_pos='end', num_caption=16):
        super().__init__()

        _tokenizer = _Tokenizer()
        n_cls = len(classnames)
        n_ctx = num_prompts
        n_cap = num_caption
        ctx_init = prompts_init
        bs = batch_size
        ctx_dim = text_model.ln_final.weight.shape[0]
        self.prompts_generator = VideoSpecificPrompt(layers=1, embed_dim=ctx_dim)
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = text_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        self.ctx = nn.Parameter(ctx_vectors)
        self.zeros_ctx = torch.zeros(n_cls, n_ctx, ctx_dim)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        learnable_prompts = [prompt_prefix + "." for name in classnames]
        learnable_tokenized_prompts = torch.cat([tokenize(p) for p in learnable_prompts])

        with torch.no_grad():
            learnable_embedding = text_model.token_embedding(learnable_tokenized_prompts)


        self.register_buffer("token_prefix", learnable_embedding[:, :1, :])

        self.register_buffer("token_suffix", learnable_embedding[:, 1 + n_ctx :, :])


        self.n_cls = n_cls
        self.n_ctx = n_ctx

        self.learnable_tokenized_prompts = learnable_tokenized_prompts

        self.name_lens = name_lens
        self.class_token_position = ctx_pos

    def forward(self, cls_text):

        ctx = self.ctx
        ctx = ctx + self.prompts_generator(ctx, cls_text.unsqueeze(1))

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class ClsPrompt(nn.Module):
    def __init__(self, classnames, text_model,):
        super().__init__()

        _tokenizer = _Tokenizer()
        n_cls = len(classnames)
        ctx_dim = text_model.ln_final.weight.shape[0]



        classnames = [name.replace("_", " ") for name in classnames]


        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        cls_prompts = [name + "." for name in classnames]        
        cls_tokenized_prompts = torch.cat([tokenize(p) for p in cls_prompts])


        with torch.no_grad():
            cls_embedding = text_model.token_embedding(cls_tokenized_prompts)


        self.register_buffer("cls_prompts", cls_embedding)  # SOS

        self.cls_tokenized_prompts = cls_tokenized_prompts


    def forward(self,):
        cls_tokenized_prompts = self.cls_tokenized_prompts
        cls_prompts = self.cls_prompts
        
        return cls_prompts, cls_tokenized_prompts



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

if __name__ == '__main__':
    with open('../classes/hmdb51_classes.txt', 'r') as f:
        classes = f.read().strip().split('\n')
    textual = CLIPTextEncoder(
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    )
    prompt_learner = TextPromptLearner(
        classnames=classes,
        text_model=textual,
        num_prompts=8,
        prompts_init='',
        CSC=True,
        ctx_pos='end'
    )
    prompts = prompt_learner()

    tokenized_prompts = prompt_learner.tokenized_prompts
    #print(tokenized_prompts.shape)
    #print(tokenized_prompts)
    #print(tokenized_prompts.argmax(dim=-1))
    text_features = textual(prompts, tokenized_prompts)
    #print(text_features.shape)