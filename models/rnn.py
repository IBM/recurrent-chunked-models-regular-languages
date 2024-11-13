#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Builders for RNN/LSTM cores."""

import torch as t
import torch.nn as nn

class ElmanRNN(nn.Module):
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 return_all_outputs: bool = False,
                 bidirectional: bool = False,
                 dropout: float = .1,
                 **kwargs
                 ):
        
        super(ElmanRNN, self).__init__()

        # The input is provided as a one-hot vector instead of token integers, so use a linear layer to select the
        # correct embedding. See `jax_transformer.py#251` for the original implementation.
        self.embedding = nn.Linear(
            input_size,
            hidden_size,
            bias=False
        )
        # TODO: make the embedding initialization scale a parameter. I coped .02 from the original paper.
        nn.init.normal_(self.embedding.weight, 0, .02)
        
        self.return_all_outputs = return_all_outputs

        self.rnn = nn.RNN(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          nonlinearity='relu',
                          bias=True,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=bidirectional)

        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, 
                x: t.Tensor
                ) -> t.Tensor:

      rnn_out = self.rnn(self.embedding(x))

      # PyTorch RNN returns (output, h_n)
      if not self.return_all_outputs:
        output = rnn_out[0][:,-1]
      else:
         output = rnn_out[0]

      return self.output_projection(output)
