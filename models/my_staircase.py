#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

import math
import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.transformer as transformer

from torch import Tensor
from enum import Enum
from .transformer_utils import TransformerNormalization

class ForwardMode(Enum):
    PARALLEL = 0
    STAIRCASE = 1


class MyStaircaseTransformer(nn.Module):
    def __init__(
        self,
        output_size: int,
        return_all_outputs: bool,
        input_size: int,
        d_model: int,
        num_heads: int,
        num_layers: Optional[int],
        group_size: int,
        num_recurrent_calls: Optional[int],
        feedforward_multiplier: int = 4,
        dropout: float = .1,
        norm_type: TransformerNormalization = TransformerNormalization.prenorm,
        is_causal: bool = False,
        diagonal: bool = False,
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

        if num_layers is not None and num_recurrent_calls is not None:
            assert num_layers / group_size == num_recurrent_calls, f"num_layers / group_size != num_recurrent_calls"
            self.num_recurrent_calls = num_recurrent_calls
        if num_layers is not None and num_recurrent_calls is None:
            assert num_layers % group_size == 0, f"num_layers ({num_layers}) must be divisible by group_size ({group_size})"
            self.num_recurrent_calls = num_layers // group_size
            print(f'Number of recurrent calls: {self.num_recurrent_calls}')
        if num_layers is None and num_recurrent_calls is not None:
            print(f'{group_size * num_recurrent_calls} layers of processing')
            self.num_recurrent_calls = num_recurrent_calls
        if num_layers is None and num_recurrent_calls is None:
            assert False, "One of num_layers or num_recurrent_calls must be set for staircase transformer"

        # The input is provided as a one-hot vector instead of token integers, so use a linear layer to select the
        # correct embedding. See `jax_transformer.py#251` for the original implementation.
        self.embedding = nn.Linear(
            input_size,
            d_model,
            bias=False
        )
        # NOTE: this uses fan_out instead of fan_in since our embedding matrix is actually a nn.Linear instead of a
        #  nn.Embedding.
        with torch.no_grad():
            nn.init.kaiming_normal_(self.embedding.weight, mode="fan_out", nonlinearity="linear")

        # Use TransformerEncoder as the recurrent unit so that it can have multiple layers in the recurrent unit.
        # Csordas et al calls this a "group", Ju et al calls this a "core".
        layer = TransformerEncoderLayerWithCache(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model*feedforward_multiplier,
            dropout=dropout,
            norm_first=norm_type == TransformerNormalization.prenorm,
            batch_first=True
        )

        # Prenorm places a final layer norm after the whole network since there is not one after the MLP residual.
        # This final norm should come at the end of the full processing, not at the end each group. Therefore, apply
        # this norm manually at the end instead of passing it to nn.TransformerEncoder() below
        if norm_type == TransformerNormalization.prenorm:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = None

        self.group = TransformerEncoderWithCache(
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

        self.mode = ForwardMode.PARALLEL
        self.diagonal = diagonal

        self.cache_steps = self.kwargs.get('staircase_cache_steps', 0)
        # Cache steps of -1 uses global caching
        if self.cache_steps == -1:
            self.cache_steps = sys.maxsize

    def forward(
        self,
        x_BLI: Tensor,
    ) -> Tensor:
        if self.mode is ForwardMode.PARALLEL:
            return self.forward_parallel(x_BLI)
        else:
            return self.forward_staircase(x_BLI)

    def forward_parallel(
        self,
        x_BLI: Tensor,
    ) -> Tensor:
        """
        B: batch
        L: length
        H: hidden dimension
        I: input size
        O: output size
        """
        x_BLH = self.embedding(x_BLI)

        L = x_BLH.size(1)
        
        if self.is_causal:
            causal_mask_LL = nn.Transformer.generate_square_subsequent_mask(L, device=x_BLH.device)
        else:
            causal_mask_LL = None

        for _ in range(self.num_recurrent_calls):
            x_BLH = self.group(x_BLH, cache=x_BLH, mask=causal_mask_LL, is_causal=self.is_causal)

        if self.final_norm:
            x_BLH = self.final_norm(x_BLH)

        out_BLO = self.output(x_BLH)
        
        return out_BLO[:, -1, :] if not self.return_all_outputs else out_BLO

    def forward_staircase(
        self, 
        x_BLI: Tensor,
    ) -> Tensor:
        """
        B: batch
        C: chunk width (also known as fix_staircase_size_forward). NOTE chunk width is dynamic. For instance, if the 
            specified chunk width is 8 and the input is length 14, the first chunk will have 8 tokens and the second
            chunk will have (14-8)=6 tokens.
        L: length
        H: hidden dimension
        I: input size
        O: output size
        N: num recurrent calls
        D: Diagonal cache tokens.
        R: Active row width measured in tokens, not chunks. This contrasts with cached row width.
          NOTE row width is dynamic and can change as forward progresses.
            For instance, the first row only has one chunk, whereas the second row has two chunks, etc.
        A: cAched width measured in tokens, not chunks. This is the gray boxes in Figure 1.
        T: Total key and value tokens. This includes tokens in the active row (R), the horizontal cache (A), and 
           the diagonal cache (D). T=R+A+D.
        """
        max_chunk_size = self.kwargs['fix_staircase_size_forward']
        # TODO: confirm that this works as expected with chunks over than size 1
        #assert chunk_size == 1, \
        #'Only a single token chunk is supported at the moment, larger chunks will require us to implement padding for '\
        #     ' the final chunk if it doesn\'t divide evenly'
        assert self.is_causal, 'Staircase only supports causal models'
        
        B, L, _ = x_BLI.shape

        # total_rows is the number of rows in the staircase diagram. Each row can be one or more transformer layers.
        # The first term calculates the number of forward chunks, and the remaining terms account for the processing
        #  necessary to process the final chunk.
        num_forward_chunks = math.ceil(L / max_chunk_size)
        total_rows = num_forward_chunks + self.num_recurrent_calls - 1

        x_BLH = self.embedding(x_BLI)

        H = x_BLH.size(-1)

        # Pre-allocate the output. This avoids having to perform torch.cat when new outputs arrive.
        output_BLH = torch.zeros((B, L, H), device=x_BLI.device)
        # output_L_offset stores the start index of the next location to write a chunk to
        output_L_offset = 0

        if self.diagonal:
            # Pre-allocate the diagonal cache. These values don't change once written, so it is more efficient to
            #  pre-allocate the block of memory instead of using torch.cat to expand the tensor during computation.
            diagonal_cache_BNLH = torch.zeros((B, self.num_recurrent_calls, L, H), device=x_BLI.device)

        x_BRH = torch.zeros((B, 0, H), device=x_BLI.device)
        cache_BAH = torch.zeros((B, 0, H), device=x_BLI.device)
        active_chunk_sizes = []
        for row_i in range(total_rows):
            # Prepare the input for this row
            new_chunk_BCH = x_BLH[:, row_i*max_chunk_size:(row_i+1)*max_chunk_size, :]
            active_chunk_sizes.append(new_chunk_BCH.size(1))

            x_BRH = torch.cat((x_BRH, new_chunk_BCH), dim=1)

            if self.diagonal:
                # Write the input tokens to the diagonal cache

                # There are three edge cases we need to account for:
                #  1. We are on the first couple of chunks, so not all of the diagonals are active yet
                #  2. We are past the final chunk but still processing the active chunks. In this case the lower diagonals
                #    need to be turned off.
                #  3. The final chunk is less than max size

                # In the earlier rows, we need to adjust the indices since not all of the diagonals will have content
                #  to write yet.
                if row_i < self.num_recurrent_calls:
                    diagonal_L_indices_min = 0
                    diagonal_N_indices_max = row_i
                else:
                    diagonal_L_indices_min = (row_i - self.num_recurrent_calls + 1) * max_chunk_size
                    diagonal_N_indices_max = self.num_recurrent_calls - 1

                diagonal_L_indices_max = (row_i + 1) * max_chunk_size
                diagonal_N_indices_min =  -1

                diagonal_L_indices = torch.arange(
                    diagonal_L_indices_min,
                    diagonal_L_indices_max,
                    device=diagonal_cache_BNLH.device
                )
                # NOTE: diagonal_N_indices goes from max to min because higher diagonals align with the lower L token
                #  indices
                diagonal_N_indices = torch.arange(
                    diagonal_N_indices_max,
                    diagonal_N_indices_min,
                    -1,
                    device=diagonal_cache_BNLH.device
                ).repeat_interleave(max_chunk_size)

                # In the later rows, we need to adjust the indices since some diagonals are no longer active.
                # This deals with the edge case where we are past the final chunk, or the final chunk is less than max
                #  chunk size. If new_chunk_BCH has size 0, then we are past the final chunk.
                if new_chunk_BCH.size(1) - max_chunk_size < 0:
                    diagonal_L_indices = diagonal_L_indices[:x_BRH.size(1)]
                    diagonal_N_indices = diagonal_N_indices[:x_BRH.size(1)]

                # Since we pre-allocate the diagonal cache, we need to write into it using Tensor.data so that the
                #  version does not change.
                diagonal_cache_BNLH.data[:, diagonal_N_indices, diagonal_L_indices] = x_BRH

                # There are no diagonal elements in the first row
                if row_i > 0:
                    # Get the diagonal elements in a flat vector. The tokens along the bottom diagonal are first,
                    #  followed by the tokens on the next diagonal up, etc.
                    diagonal_mask_rows_1N11 = torch.arange(self.num_recurrent_calls, device=x_BRH.device)[None, :, None, None]
                    diagonal_mask_row_offset_BNLH = torch.full(diagonal_cache_BNLH.shape, row_i, device=x_BRH.device)
                    diagonal_mask_max_L_index_BNLH = (diagonal_mask_row_offset_BNLH - diagonal_mask_rows_1N11) * max_chunk_size
                    

                    diagonal_mask_BNLH = torch.arange(L, device=x_BRH.device)[None, None, :, None] < diagonal_mask_max_L_index_BNLH

                    diagonal_mask_min_N_index = max(0, row_i + self.num_recurrent_calls - total_rows)
                    diagonal_mask_min_N_mask = torch.arange(self.num_recurrent_calls, device=x_BRH.device)[None, :, None, None] >= diagonal_mask_min_N_index

                    diagonal_mask_BNLH = torch.logical_and(diagonal_mask_BNLH, diagonal_mask_min_N_mask)

                    # TODO: I think that masked_select causes a GPU-CPU synchronization. If we ever want to scale this up
                    #  it's worth looking into whether we can figure out a solution to avoid synchronization
                    diagonal_cache_tokens_BDH = diagonal_cache_BNLH.masked_select(diagonal_mask_BNLH).view(B, -1, H)

                    # TODO: using diagonals with masking ends up with a huge computational cost of (L*L*N). This is because
                    #  we compute the attention between each token and other token both in a specific layer and across
                    #  all other layers. We also end up with a very sparse attention mask. I think that pytorch has an
                    #  attention context manager for this scenario. If that doens't work, look into the xformers library
                    #  which also has mask-aware attention implementations.


                    # The active tokens in x_BRH should only be able to attend to their own diagonal. Since the diagonal
                    # cache is flattened into a vector, this makes the masking a bit complicated. The most recent chunk
                    # x_BRH[:, -1, :] should attend to the first row_i chunks in diagonal_cache_tokens_BDH, then 
                    # the second most recent chunk x_BRH[:, -2, :] should attend to the next row_i-1 chunks, etc.
                    # This masking scheme can be accomplished by using a cumulative sum to get the offsets for the 
                    # decreasing visibility each active chunk has going from right to left.
                    diagonal_attention_mask_RD = diagonal_cache_tokens_BDH.new_full(
                        (x_BRH.size(1), diagonal_cache_tokens_BDH.size(1)),
                        -math.inf
                    )

                    # When we are finishing the final active chunks, we need to ignore the earlier diagonals since we 
                    #  are not processing new chunks which should look at the first diagonal
                    diagonal_flat_indices_D = torch.arange(diagonal_attention_mask_RD.size(1))

                    deactivated_diagonals = diagonal_mask_min_N_index

                    chunk_indices_offset = torch.cumsum(
                        torch.cat((torch.zeros(1), torch.arange(row_i - deactivated_diagonals, -1, -1))),
                        dim=0,
                        dtype=torch.int64
                    )

                    diagonal_attention_mask_min_chunk_index = chunk_indices_offset[0:self.num_recurrent_calls, None]
                    diagonal_attention_mask_max_chunk_index = chunk_indices_offset[1:self.num_recurrent_calls+1, None]

                    # We need to convert from chunks to token indices
                    # First, convert the chunk index into a token index by multiplying by the chunk size.
                    # Second, the mask needs to be repeated for the number of query tokens which is equal to the row size.
                    diagonal_attention_mask_min_token_index = diagonal_attention_mask_min_chunk_index * max_chunk_size
                    diagonal_attention_mask_max_token_index = diagonal_attention_mask_max_chunk_index * max_chunk_size

                    # On the last row, there is no diagonal for the empty chunk
                    if row_i == total_rows - 1:
                        diagonal_attention_mask_min_token_index = diagonal_attention_mask_min_token_index.repeat_interleave(torch.tensor(list(reversed(active_chunk_sizes[:-1]))), dim=0)
                        diagonal_attention_mask_max_token_index = diagonal_attention_mask_max_token_index.repeat_interleave(torch.tensor(list(reversed(active_chunk_sizes[:-1]))), dim=0)
                    else:
                        diagonal_attention_mask_min_token_index = diagonal_attention_mask_min_token_index.repeat_interleave(torch.tensor(list(reversed(active_chunk_sizes))), dim=0)
                        diagonal_attention_mask_max_token_index = diagonal_attention_mask_max_token_index.repeat_interleave(torch.tensor(list(reversed(active_chunk_sizes))), dim=0)

                    # In early rows, we only want to look at the first R diagonals
                    # At the final rows, we only want to look at the last R diagonals
                    # TODO does this if statement do anything anymore?
                    if row_i < self.num_recurrent_calls or row_i + self.num_recurrent_calls > total_rows:
                        diagonal_attention_mask_min_token_index = diagonal_attention_mask_min_token_index[:x_BRH.size(1)]
                        diagonal_attention_mask_max_token_index = diagonal_attention_mask_max_token_index[:x_BRH.size(1)]

                    diagonal_attention_mask_min_token_index = diagonal_attention_mask_min_token_index.flip(dims=(0,))
                    diagonal_attention_mask_max_token_index = diagonal_attention_mask_max_token_index.flip(dims=(0,))

                    unmasked_values_RD = torch.logical_and(
                        diagonal_attention_mask_min_token_index <= diagonal_flat_indices_D,
                        diagonal_attention_mask_max_token_index > diagonal_flat_indices_D
                    )
                    diagonal_attention_mask_RD[unmasked_values_RD] = 0

            # Generate the attention mask
            causal_mask_RR = nn.Transformer.generate_square_subsequent_mask(x_BRH.size(1), device=x_BRH.device)
            # Anything in the cache can be attended to
            cache_mask_RA = causal_mask_RR.new_zeros((causal_mask_RR.size(0), cache_BAH.size(1)))

            #bias_towards_recurrence = False
            #if bias_towards_recurrence and row_i != 0 and row_i != total_rows - 1:
            #    causal_mask_RR[-1, 0] = 2

            if self.diagonal and row_i > 0:
                # The keys and queries should include the cache and the `x` elements themselves
                # NOTE: the ordering of the arguments in this torch.cat call must match the ordering of the arguments in
                #   the torch.cat call below that creates the mask
                cache_and_x_BTH = torch.cat((diagonal_cache_tokens_BDH, cache_BAH, x_BRH), dim=1)
                # NOTE: the ordering of the arguments in this torch.cat call must match the ordering of the arguments in
                #   the torch.cat call above that creates the key and value vectors
                cache_and_causal_mask_RT = torch.cat((diagonal_attention_mask_RD, cache_mask_RA, causal_mask_RR), dim=1)
            else:
                # The keys and queries should include the cache and the `x` elements themselves
                # NOTE: the ordering of the arguments in this torch.cat call must match the ordering of the arguments in
                #   the torch.cat call below that creates the mask
                cache_and_x_BTH = torch.cat((cache_BAH, x_BRH), dim=1)
                # NOTE: the ordering of the arguments in this torch.cat call must match the ordering of the arguments in
                #   the torch.cat call above that creates the key and value vectors
                cache_and_causal_mask_RT = torch.cat((cache_mask_RA, causal_mask_RR), dim=1)

            # Using a cache makes attention non-causal, I think. I don't quite understand what it means for cross
            # attention to be causal....
            x_BRH = self.group(
                x_BRH,
                cache_and_x_BTH,
                mask=cache_and_causal_mask_RT,
                is_causal=self.cache_steps == 0 and not self.diagonal
            )

            # Store the most oldest chunk if it has finished processing
            if row_i >= (self.num_recurrent_calls - 1):
                active_chunk_size = active_chunk_sizes[0]
                output_BLH[:, output_L_offset:output_L_offset+active_chunk_size, :] = x_BRH[:, :active_chunk_size, :]
                output_L_offset += active_chunk_size
                active_chunk_sizes = active_chunk_sizes[1:]

                if self.cache_steps:
                    if row_i - self.num_recurrent_calls + 1 >= self.cache_steps:
                        # Remove the oldest cached chunk
                        cache_BAH = cache_BAH[:, max_chunk_size:, :]
                    # Add the newly finished chunk to the cache
                    cache_BAH = torch.cat((cache_BAH, x_BRH[:, :active_chunk_size, :]), dim=1)

                # Remove the newly finished chunk from the active chunks
                x_BRH = x_BRH[:, active_chunk_size:, :]
                   
        if self.final_norm:
            output_BLH = self.final_norm(output_BLH)

        out_BLO = self.output(output_BLH)
        
        return out_BLO[:, -1, :] if not self.return_all_outputs else out_BLO


class TransformerEncoderWithCache(transformer.TransformerEncoder):
    def forward(
        self,
        src: Tensor,
        cache: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None) -> Tensor:

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first

        seq_len = transformer._get_seq_len(src, batch_first)
        is_causal = transformer._detect_is_causal_mask(mask, is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, cache, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayerWithCache(transformer.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Turn dropout off in attention
        self.self_attn.dropout = 0

    def forward(
        self,
        src: Tensor,
        cache: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False
    ) -> Tensor:
        r"""Override the forward function to support passing cache for keys and values during self attention
        """

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        # TODO: what does a causal memory mask even mean for attention?
        if self.norm_first:
            x = x + self._sa_block_with_cache(self.norm1(x), self.norm1(cache), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block_with_cache(x, cache, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x
    
    def _sa_block_with_cache(
        self,
        x: Tensor,
        cache: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        need_weights=False
    ) -> Tensor:
        x, weights = self.self_attn(x, cache, cache,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights, is_causal=is_causal,
                    average_attn_weights=False)
        return self.dropout1(x)


PRETRAINED_FULL_DROPOUT_CHECKPOINTS = [
    'out/my_staircase/parity_check/my_staircase_parity_check_params_263016_layers_4_embed_104_state_256_LR_0.0001_WD_0.0_trainstep_1000000_trainlen_40_SEED_239748746.pt',
    'out/my_staircase/parity_check/my_staircase_parity_check_params_263016_layers_4_embed_104_state_256_LR_0.0001_WD_0.0_trainstep_1000000_trainlen_40_SEED_219576059.pt',
    'out/my_staircase/parity_check/my_staircase_parity_check_params_263016_layers_4_embed_104_state_256_LR_0.0001_WD_0.0_trainstep_1000000_trainlen_40_SEED_3980338655.pt'
]

PRETRAINED_NO_ATTN_DROPOUT_CHECKPOINTS = [
    'out/my_staircase/parity_check/my_staircase_parity_check_params_263016_layers_4_embed_104_state_256_LR_0.0001_WD_0.0_trainstep_1000000_trainlen_40_SEED_4259586903.pt',
    'out/my_staircase/parity_check/my_staircase_parity_check_params_263016_layers_4_embed_104_state_256_LR_0.0001_WD_0.0_trainstep_1000000_trainlen_40_SEED_727400119.pt',
    #'out/my_staircase/parity_check/my_staircase_parity_check_params_263016_layers_4_embed_104_state_256_LR_0.0001_WD_0.0_trainstep_1000000_trainlen_40_SEED_3401764276.pt',
]

PRETRAINED_NO_DROPOUT_CHECKPOINTS = [
    'out/my_staircase/parity_check/my_staircase_parity_check_params_263016_layers_4_embed_104_state_256_LR_0.0001_WD_0.0_trainstep_1000000_trainlen_40_SEED_2609885964.pt',
    'out/my_staircase/parity_check/my_staircase_parity_check_params_263016_layers_4_embed_104_state_256_LR_0.0001_WD_0.0_trainstep_1000000_trainlen_40_SEED_3313954355.pt',
    'out/my_staircase/parity_check/my_staircase_parity_check_params_263016_layers_4_embed_104_state_256_LR_0.0001_WD_0.0_trainstep_1000000_trainlen_40_SEED_3434780149.pt',
]

PRETRAINED_CHECKPOINTS = PRETRAINED_NO_ATTN_DROPOUT_CHECKPOINTS