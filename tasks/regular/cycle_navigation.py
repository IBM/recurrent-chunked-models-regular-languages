#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Compute the final state after randomly walking on a circle."""

import torch as t
import torch.nn.functional as F

from recurrent_chunked_models_regular_languages.tasks import task


class CycleNavigation(task.GeneralizationTask):
  """A task with the goal of computing the final state on a circle.

  The input is a string of actions, composed of 0s, 1s or -1s. The actions give
  directions to take on a finite length circle (0 is for stay, 1 is for right,
  -1 is for left). The goal is to give the final position on the circle after
  all the actions have been taken. The agent starts at position 0.

  By default, the length the circle is 5.

  Examples:
    1 -1 0 -1 -1 -> -2 = class 3
    1 1 1 -1 -> 2 = class 2

  Note that the sampling is jittable so it is fast.
  """

  @property
  def _cycle_length(self) -> int:
    """Returns the cycle length, number of possible states."""
    return 5

  def sample_batch(self, batch_size: int,
                   length: int) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    actions = t.randint(
        size=(batch_size, length), low=0, high=3)
    final_states = t.sum(actions - 1, axis=1) % self._cycle_length
    final_states = F.one_hot(final_states, num_classes=self.output_size)
    one_hot_strings = F.one_hot(actions, num_classes=self.input_size)
    return {"input": one_hot_strings, "output": final_states}

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 3

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return self._cycle_length
