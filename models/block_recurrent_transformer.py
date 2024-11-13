#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import torch
import torch.nn as nn
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer

class BlockRecurrentTransformerWrapper(nn.Module):
    def __init__(
        self,
        output_size: int,
        return_all_outputs: bool,
        input_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        chunk_size: int,
        num_state_vectors: int,
        recurrent_layer_index: int,
        **kwargs
    ):
        super().__init__()
        self.model = BlockRecurrentTransformer(
            num_tokens = 1, # vocab size, this will be written over below in this __init__ call
            dim = d_model,                      # model dimensions
            depth = num_layers,                      # depth
            dim_head = d_model // num_heads,                  # attention head dimensions
            heads = num_heads,                      # number of attention heads
            max_seq_len = kwargs.get('max_sequence_length', 512),             # the total receptive field of the transformer, in the paper this was 2 * block size
            block_width = chunk_size,              # block size - total receptive field is max_seq_len, 2 * block size in paper. the block furthest forwards becomes the new cached xl memories, which is a block size of 1 (please open an issue if i am wrong)
            num_state_vectors = num_state_vectors,        # number of state vectors, i believe this was a single block size in the paper, but can be any amount
            recurrent_layers = (recurrent_layer_index,),        # where to place the recurrent layer(s) for states with fixed simple gating
            use_compressed_mem = False,     # whether to use compressed memories of a single block width, from https://arxiv.org/abs/1911.05507
            compressed_mem_factor = 4,      # compression factor of compressed memories
            use_flash_attn = True,           # use flash attention, if on pytorch 2.0,
        )

        # The input is provided as a one-hot vector instead of token integers, so use a linear layer to select the
        # correct embedding. See `jax_transformer.py#251` for the original implementation.
        self.model.token_emb = nn.Linear(
            input_size,
            d_model,
            bias=False
        )
        # NOTE: this uses fan_out instead of fan_in since our embedding matrix is actually a nn.Linear instead of a
        #  nn.Embedding.
        with torch.no_grad():
            nn.init.kaiming_normal_(self.model.token_emb.weight, mode="fan_out", nonlinearity="linear")
            # Match the magnitude between the initial state ID embeddings and the token embeddings. Note that this
            #  uses fan_in while the token_emb use fan_out, this is due to initialization via nn.Linear vs nn.Parameter.
            nn.init.kaiming_normal_(self.model.layers[recurrent_layer_index-1][0].state_container.init_state, mode="fan_in", nonlinearity="linear")

        self.model.to_logits[1] = nn.Linear(
            d_model,
            output_size,
            bias=False
        )

        self.return_all_outputs = return_all_outputs


    def forward(
        self,
        x_BLI: torch.Tensor,
    ):
        out_BLO, _, _ = self.model(
            x_BLI,
            return_loss=False,
            return_memories_and_states=False
        )
        return out_BLO[:, -1, :] if not self.return_all_outputs else out_BLO
