#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Compute whether the number of 01's and 10's is even."""

from jax import numpy as jnp

import torch as t
import torch.nn.functional as F

from recurrent_chunked_models_regular_languages.tasks import task


class EvenPairs(task.GeneralizationTask):
  """A task with the goal of checking whether the number of 01s and 10s is even.

  The input is a binary string, composed of 0s and 1s. If the result is even,
  the class is 0, otherwise it's one.

  Examples:
    001110 -> 1 '10' and 1 '01' -> class 0
    0101001 -> 2 '10' and 3 '01' -> class 1

  Note the sampling is jittable so this task is fast.
  """

  # @functools.partial(jax.jit, static_argnums=(0, 2, 3))
  def sample_batch(self, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    strings = t.randint(
        low=0,
        high=2,
        size=(batch_size, length),
    )
    one_hot_strings = F.one_hot(strings, num_classes=2)
    unequal_pairs = t.logical_xor(strings[:, :-1], strings[:, 1:])
    odd_unequal_pairs = t.sum(unequal_pairs, axis=-1) % 2
    return {
        'input': one_hot_strings,
        'output': F.one_hot(odd_unequal_pairs, num_classes=self.output_size),
    }

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
