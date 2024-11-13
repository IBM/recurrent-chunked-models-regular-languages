#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Evaluation of a network on sequences of different lengths."""

import dataclasses
import random
from typing import Any, Callable, Mapping

from absl import logging
import numpy as np
import tqdm

import torch.nn as nn
import torch as t

import recurrent_chunked_models_regular_languages.wandb_wrapper as wandb

from recurrent_chunked_models_regular_languages.experiments import utils
from recurrent_chunked_models_regular_languages.tasks import task as task_lib


_Batch = Mapping[str, t.Tensor]

device = 'cuda' if t.cuda.is_available() else 'cpu'


@dataclasses.dataclass
class EvaluationParams:
  """The parameters used for range evaluation of networks."""
  #model: hk.Transformed
  #params: hk.Params
  model: nn.Module
  task: task_lib.GeneralizationTask

  single_output: bool

  accuracy_fn: Callable[[t.Tensor, t.Tensor], t.Tensor]
  sample_batch: Callable[[t.Tensor, int, int], _Batch]

  max_test_length: int
  total_batch_size: int
  sub_batch_size: int  # We use this to avoid memory overflow.

  is_autoregressive: bool = False

  computation_steps_mult: int = 0

  use_wandb: bool = False
  include_eos: bool = True


def range_evaluation(
    eval_params: EvaluationParams,
    use_tqdm: bool = False,
    tboard_writer = None
) -> list[Mapping[str, Any]]:
  """Evaluates the model on longer, never seen strings and log the results.

  Args:
    eval_params: The evaluation parameters, see above.
    use_tqdm: Whether to use a progress bar with tqdm.

  Returns:
    The list of dicts containing the accuracies.
  """

  model = eval_params.model
  #params = eval_params.params
  # TODO: why does turning off dropout hurt the model's IID performance so much?
  model.eval()

  writer = tboard_writer

  random.seed(1)
  np.random.seed(1)
  t.manual_seed(1)

  results = []
  lengths = range(1, eval_params.max_test_length + 1)
  with t.inference_mode():
    if use_tqdm:
      lengths = tqdm.tqdm(lengths)
    for length in lengths:

      output_length = eval_params.task.output_length(length)
      # We need to clear the cache of jitted functions, to avoid overflow as we
      # are jitting len(lengths) ones, which can be a lot.
      #apply_fn.clear_cache()
      sub_accuracies = []
      for _ in range(eval_params.total_batch_size // eval_params.sub_batch_size):
        batch = eval_params.sample_batch(eval_params.sub_batch_size, length)

        batch_input = batch['input']
        batch_output = batch['output']
        # TODO: Find a nicer way to go around this
        # Sequence padding function
        if eval_params.is_autoregressive:
            raise ValueError("Autoregressive mode is not supported at the moment. Date: 24.01.2024.")
        else:
          pad_sequence = utils.pad_sequence_with_empty_targets(
            generalization_task=eval_params.task,
            computation_steps_mult=eval_params.computation_steps_mult,
            include_eos=eval_params.include_eos
        )
          
        if eval_params.is_autoregressive:
          raise ValueError("Autoregressive mode is not supported at the moment. Date: 24.01.2024.")
        else:
          batch_input = pad_sequence(batch_input)

        batch_input = batch_input.to(device)
        batch_output = batch_output.to(device)

        if eval_params.is_autoregressive:
          raise ValueError("Autoregressive mode is not supported at the moment. Date: 24.01.2024.")
        else:
          outputs = model(batch_input)
        
        if type(outputs) is dict:
          reg_loss = outputs['reg_loss']
          outputs = outputs['output']

        if not eval_params.single_output:
          outputs = outputs[:, -output_length:]

        sub_accuracies.append(
            float(t.mean(eval_params.accuracy_fn(outputs, batch_output))))
      log_data = {
          'length': length,
          'accuracy': np.mean(sub_accuracies),
      }

      if eval_params.use_wandb:
        wandb.log(
          {
            'eval': {
              'length': length,
              'accuracy': np.mean(sub_accuracies),
            },
          },
        )
        pass# TODO

      if writer:
        writer.add_scalar("Accuracy/oodlen", np.mean(sub_accuracies), length)

      logging.info(log_data)
      results.append(log_data)
  return results
