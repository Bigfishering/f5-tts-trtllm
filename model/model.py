"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)
import math
import numpy as np
import torch
from torch import nn
import tensorrt as trt
from collections import OrderedDict
from ..._utils import str_dtype_to_trt, trt_dtype_to_str, trt_dtype_to_np
from ...plugin import current_all_reduce_helper
from ..modeling_utils import PretrainedConfig, PretrainedModel
from ...functional import (Tensor, allgather, arange, chunk, concat, constant,
                           cos, exp, expand, shape, silu, sin, slice, split,
                           unsqueeze, squeeze, cast)
from ...module import Module, ModuleList
from ...layers import Linear

from .modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis, get_pos_embed_indices,
)

# Text embedding
class TextEmbedding(Module):
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

class InputEmbedding(Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim = out_dim)

    def forward(self, x: float['b n d'], cond: float['b n d'], drop_audio_cond = False):
        # if drop_audio_cond:  # cfg for cond audio
        x = self.proj(concat([x, cond], dim = -1))
        return self.conv_pos_embed(x) + x
    
# Transformer backbone using DiT blocks
class F5TTS(PretrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.f5_transformer = DiT_transformer(config)
        self.dtype = str_dtype_to_trt(config.dtype)

    def forward(self,
                noise: float['b n d'],  # nosied input audio
                cond: float['b n d'],  # masked cond audio
                cond_drop: float['b n d'],
                time: float['b n'],  # time step
                rope_cos: float['b n d'],
                rope_sin: float['b n d'],
                t_scale: float['b'],
                mask: bool['b n'] | None = None):
        
        pred = self.f5_transformer(x = noise, cond = cond, cond_drop = cond_drop, time = time, rope_cos = rope_cos, rope_sin = rope_sin)
        pred, pred1 = chunk(pred, 2, dim = 0)
        twos = constant(np.array([2], dtype = np.float32)).cast(noise.dtype)

        noise = noise + (pred + (pred - pred1) * twos) * t_scale
        noise.mark_output('denoised', self.dtype)
        return noise

    def prepare_inputs(self, **kwargs):
        mapping = self.config.mapping
        if mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(mapping, 1)

        noise = Tensor(
            name='noise',
            dtype=self.dtype,
            shape=[1, -1, 100],
            dim_range=OrderedDict([
                ('batch_size', [[1]*3]),
                ('max_duratuion', [[100, 1226, 2000]]),
                ('n_mels', [[100]*3]),
            ]))
        cond = Tensor(
            name='cond',
            dtype=self.dtype,
            shape=[1, -1, 612],
            dim_range=OrderedDict([
                ('batch_size', [[1]*3]),
                ('max_duratuion', [[100, 1226, 2000]]),
                ('embeded_length', [[612]*3]),
        ]))
        cond_drop = Tensor(
            name='cond_drop',
            dtype=self.dtype,
            shape=[1, -1, 612],
            dim_range=OrderedDict([
                ('batch_size', [[1]*3]),
                ('max_duratuion', [[100, 1226, 2000]]),
                ('embeded_length', [[612]*3]),
        ]))
        time = Tensor(name='time',
                            dtype=self.dtype,
                            shape=[1, 256],
                            dim_range=OrderedDict([
                                ('batch_size_t', [[1]*3]),
                                ('freq_dim', [[256]*3]),
                            ]))
        rope_cos = Tensor(name='rope_cos',
                            dtype=self.dtype,
                            shape=[1, -1, 64],
                            dim_range=OrderedDict([
                                ('batch_size', [[1]*3]),
                                ('max_duratuion', [[100, 1226, 2000]]),
                                ('head_dim', [[64]*3]),
                            ]))
        rope_sin = Tensor(name='rope_sin',
                            dtype=self.dtype,
                            shape=[1, -1, 64],
                            dim_range=OrderedDict([
                                ('batch_size', [[1]*3]),
                                ('max_duratuion', [[100, 1226, 2000]]),
                                ('head_dim', [[64]*3]),
                            ]))
        t_scale = Tensor(name='t_scale',
                            dtype=self.dtype,
                            shape=[1],
                            dim_range=OrderedDict([
                                ('diff_t', [1])
                            ]))
        return {'noise': noise, 'cond': cond, 'cond_drop': cond_drop, 'time': time, 'rope_cos': rope_cos, 'rope_sin': rope_sin, 't_scale': t_scale}

class DiT_transformer(PretrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.time_embed = TimestepEmbedding(config.hidden_size) # √
        if config.text_dim is None:
            text_dim = config.mel_dim
        self.input_embed = InputEmbedding(config.mel_dim, config.text_dim, config.hidden_size)

        self.dim = config.hidden_size
        self.depth = config.num_hidden_layers
        self.transformer_blocks = ModuleList(
            [
                DiTBlock(
                    dim = self.dim,
                    heads = config.num_attention_heads,
                    dim_head = config.dim_head,
                    ff_mult = config.ff_mult,
                    dropout = config.dropout
                )
                for _ in range(self.depth)
            ]
        )
        
        self.norm_out = AdaLayerNormZero_Final(config.hidden_size)  # final modulation
        self.proj_out = Linear(config.hidden_size, config.mel_dim)

    def forward(
            self,
            x: float['b n d'],  # nosied input audio
            cond: float['b n d'],  # masked cond audio
            cond_drop: float['b n d'],
            time: float['b n'],  # time step
            rope_cos: float['b n d'] ,
            rope_sin: float['b n d'],
            mask: bool['b n'] | None = None,
            scale = 1.0
    ):
        t = self.time_embed(time)
        x = concat([self.input_embed(x, cond), self.input_embed(x, cond_drop)], dim = 0)
        
        for block in self.transformer_blocks:
            x = block(x, t, rope_cos = rope_cos, rope_sin = rope_sin, mask=mask, scale = scale)
        return self.proj_out(self.norm_out(x, t))
