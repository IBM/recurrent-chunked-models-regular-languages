#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
import torch.nn as nn

from torch import Tensor
from .transformer_utils import TransformerNormalization


class Transformer(nn.Module):
    def __init__(
        self,
        output_size: int,
        return_all_outputs: bool,
        input_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_multiplier: int = 4,
        dropout: float = .1,
        norm_type: TransformerNormalization = TransformerNormalization.prenorm,
        is_causal: bool = False,
        **kwargs
    ) -> None:
        r"""
        Args:
            num_layers (`int`): The number of layers.
            num_heads (`int`): The number of attention heads.
        """
        super().__init__()

        assert norm_type is not TransformerNormalization.perinorm, 'Perinorm is not implemented for Transformer yet.'

        # The input is provided as a one-hot vector instead of token integers, so use a linear layer to select the
        # correct embedding. See `jax_transformer.py#251` for the original implementation.
        self.embedding = nn.Linear(
            input_size,
            d_model,
            bias=False
        )
        # TODO: make the embedding initialization scale a parameter. I coped .02 from the original paper.
        nn.init.normal_(self.embedding.weight, 0, .02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model*feedforward_multiplier,
            dropout=dropout,
            norm_first=norm_type == TransformerNormalization.prenorm,
            batch_first=True
        )
        # Turn off dropout on the attention computation
        layer.self_attn.dropout=0

        if norm_type == TransformerNormalization.prenorm:
            final_norm = nn.LayerNorm(d_model)
        else:
            final_norm = None

        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            norm=final_norm
        )

        self.output = nn.Linear(
            d_model,
            output_size,
            bias=False
        )

        self.return_all_outputs = return_all_outputs
        self.is_causal = is_causal


    def forward(
        self, 
        x_BLI: Tensor,
    ) -> Tensor:
        """
        B: batch
        L: length
        D: dimension
        I: input size
        O: output size
        """
        x_BLD = self.embedding(x_BLI)

        L = x_BLD.size(1)

        if self.is_causal:
            causal_mask_LL = nn.Transformer.generate_square_subsequent_mask(L, device=x_BLI.device)
        else:
            causal_mask_LL = None

        x_BLD = self.transformer(x_BLD, mask=causal_mask_LL, is_causal=self.is_causal)

        out_BLO = self.output(x_BLD)

        return out_BLO[:, -1, :] if not self.return_all_outputs else out_BLO
