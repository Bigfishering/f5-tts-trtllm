"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis, get_pos_embed_indices,
)


# Text embedding

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers = 0, conv_mult = 2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def forward(self, text: int['b nt'], seq_len):
        text = self.text_embed(text) # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            pos_idx = get_pos_embed_indices(torch.zeros(1, dtype=torch.int32), seq_len, max_pos=self.precompute_max_pos)
            # convnextv2 blocks
            text = self.text_blocks(text + self.freqs_cis[pos_idx])

        return text


# noised input audio and context mixing embedding

class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim = out_dim)

    def forward(self, x: float['b n d'], cond: float['b n d'], drop_audio_cond = False):
        # if drop_audio_cond:  # cfg for cond audio
        x = self.proj(torch.cat((x, cond), dim = -1))
        return self.conv_pos_embed(x) + x
    

# Transformer backbone using DiT blocks

class DiT(nn.Module):
    def __init__(self, *, 
                 dim, depth = 8, heads = 8, dim_head = 64, dropout = 0.1, ff_mult = 4,
                 mel_dim = 100, text_num_embeds = 256, text_dim = None, conv_layers = 0,
                 long_skip_connection = False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers = conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth
        
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    ff_mult = ff_mult,
                    dropout = dropout
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias = False) if long_skip_connection else None
        
        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
            self,
            x: float['b n d'],  # nosied input audio
            cond: float['b n d'],  # masked cond audio
            cond_drop,
            time: float['b'] | float[''],  # time step
            rope_cos,
            rope_sin,
            qk_rotated_empty,
            mask: bool['b n'] | None = None
    ):
        t = self.time_embed(time)
        x = torch.cat((self.input_embed(x, cond), self.input_embed(x, cond_drop)), dim=0)
        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin, qk_rotated_empty=qk_rotated_empty)
        return self.proj_out(self.norm_out(x, t))
