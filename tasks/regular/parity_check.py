#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Compute whether the number of 1s in a string is even."""

import torch as t
import torch.nn.functional as F

from recurrent_chunked_models_regular_languages.tasks import task

class ParityCheck(task.GeneralizationTask):
  """A task with the goal of counting the number of '1' in a string, modulo 2.

  The input is a string, composed of 0s and 1s. If the result is even, the class
  is 0, otherwise it's 1.

  Examples:
    1010100 -> 3 1s (odd) -> class 1
    01111 -> 4 1s (even) -> class 0

  Note that the sampling is jittable so this task is fast.
  """

  def sample_batch(self, batch_size: int, length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    strings = t.randint(low=0, high=2, size=(batch_size, length))
    n_b = t.sum(strings, axis=1) % 2
    n_b = F.one_hot(n_b, num_classes = 2)
    one_hot_strings = F.one_hot(strings, num_classes = 2)
    return {"input": one_hot_strings, "output": n_b}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
