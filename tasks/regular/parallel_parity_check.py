#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Parallel parity automata"""

import torch as t
import torch.nn.functional as F

from recurrent_chunked_models_regular_languages.tasks import task

class ParallelParityCheck(task.GeneralizationTask):

  def __init__(self,
               n=4,                     # The number of parallel automata 
               explicit_tokens=True     # If True, the inputs for all parallel FSAs are encoded using different vocabularies.
               ):
    
    # Number of parallel automata
    self.n = n

    # Are the tokens corresponding to different machines encoded differently?
    self.explicit_tokens = explicit_tokens

    # Used for converting input encodings from implicit to explicit.
    # Implicit = The inputs controlling different FSAs come from the same vocabulary
    # Explicit = The inputs controlling different FSAs come from different vocabularies
    self.explicit_tokens_vector = t.Tensor(list([[2*i] for i in range(n)])).flatten()

    # Used for converting vectors representing binary numbers into base-10 numbers
    self.binary_number_vector = t.Tensor(list([2**i] for i in range(n))).squeeze(-1)

  def sample_batch(self, batch_size: int, length: int):
    """Returns a batch of strings and the expected class."""

    # Inputs B x N x L
    # B: Batch size
    # N: Number of parallel FSAs
    # L: Sequence length
    strings = t.randint(low=0, high=2, size=(batch_size, self.n, length))

    # Final automaton states
    # B x N
    n_b = t.sum(strings, axis=-1) % 2

    # Converting final automaton states into a single number representing the joint final state
    # B x 1
    n_b = t.sum(n_b * self.binary_number_vector, axis=-1).long()
    n_b = F.one_hot(n_b, num_classes = 2**self.n)

    # Converting implicit input form to explicit input form
    if self.explicit_tokens:
      strings = (strings.transpose(-1, -2) + self.explicit_tokens_vector).transpose(-1, -2)

    # Flatten the input strings
    # The flattening is done s.t. the input sequence effectively consists of sequential chunks of length N, each chunk containing one symbol for each FSA.
    # We effectively cycle through the FSAs, where each successive input controlls a successive FSA
    strings = strings.transpose(-1,-2).reshape(batch_size, length*self.n).long()

    # One hot encoding the inputs
    one_hot_strings = F.one_hot(strings, num_classes = 2*self.n if self.explicit_tokens else 2)

    return {"input": one_hot_strings, "output": n_b}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2*self.n if self.explicit_tokens else 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2**self.n