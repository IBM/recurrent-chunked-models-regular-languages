#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
import torch.nn as nn

from torch import Tensor
from .transformer_utils import TransformerNormalization


class UniversalTransformer(nn.Module):
    def __init__(
        self,
        output_size: int,
        return_all_outputs: bool,
        input_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        group_size: int,
        feedforward_multiplier: int = 4,
        dropout: float = .1,
        norm_type: TransformerNormalization = TransformerNormalization.prenorm,
        is_causal: bool = False,
        **kwargs
    ) -> None:
        r"""
        Args:
            num_layers (`int`): The number of layers. This corresponds to the total number of times that a token is
                passed through a Transformer block. The number of unique Transformer blocks is calculated using 
                num_layers and group_size.
            num_heads (`int`): The number of attention heads.
            group_size (`int`): The number of unique Transformer layers (attention+feedforward) in a group.
                The parameters in the group ARE NOT shared.
        """
        super().__init__()

        assert norm_type is not TransformerNormalization.perinorm, 'Perinorm is not implemented for UT yet.'

        assert num_layers % group_size == 0, f"num_layers ({num_layers}) must be divisible by group_size ({group_size})"

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

        # Use TransformerEncoder as the recurrent unit so that it can have multiple layers in the recurrent unit.
        # Csordas et al calls this a "group", Ju et al calls this a "core".
        self.num_recurrent_calls = num_layers // group_size
        print(f'Number of recurrent calls: {self.num_recurrent_calls}')

        # Prenorm places a final layer norm after the whole network since there is not one after the MLP residual.
        # This final norm should come at the end of the full processing, not at the end each group. Therefore, apply
        # this norm manually at the end instead of passing it to nn.TransformerEncoder() below
        if norm_type == TransformerNormalization.prenorm:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = None

        # TODO: confirm for a group with multiple layers that the layer parameters are independent.
        self.group = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=group_size,
        )

        self.output_size = output_size
        self.output = nn.Linear(
            d_model,
            output_size,
            bias=False
        )

        self.return_all_outputs = return_all_outputs
        self.is_causal = is_causal
        self.kwargs = kwargs


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
            causal_mask_LL = nn.Transformer.generate_square_subsequent_mask(L)
        else:
            causal_mask_LL = None

        for recurrent_call_i in range(self.num_recurrent_calls):
            x_BLD = self.group(x_BLD, mask=causal_mask_LL)

        if self.final_norm:
            x_BLD = self.final_norm(x_BLD)

        out_BLO = self.output(x_BLD)
        
        return out_BLO[:, -1, :] if not self.return_all_outputs else out_BLO
