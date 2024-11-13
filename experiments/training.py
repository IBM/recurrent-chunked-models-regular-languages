#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Training loop for length generalization experiments."""

import dataclasses
import sys
from typing import Any, Callable, Mapping, Optional, List

import torch as t
import torch.nn as nn

import numpy as np
import tqdm
import recurrent_chunked_models_regular_languages.wandb_wrapper as wandb

from recurrent_chunked_models_regular_languages.experiments import curriculum as curriculum_lib
from recurrent_chunked_models_regular_languages.experiments import range_evaluation
from recurrent_chunked_models_regular_languages.tasks import task as task_lib
from recurrent_chunked_models_regular_languages.experiments import utils
from recurrent_chunked_models_regular_languages.models import my_staircase

from torch.utils.tensorboard import SummaryWriter


# import wandb
# wandb.login()

#_LossMetrics = Optional[Mapping[str, jnp.ndarray]]
_LossMetrics = Optional[Mapping[str, t.Tensor]]
# #_LossFn = Callable[[chex.Array, chex.Array], tuple[float, _LossMetrics]]
# _LossFn = Callable[[t.Tensor, t.Tensor], tuple[float, _LossMetrics]]
# #_AccuracyFn = Callable[[chex.Array, chex.Array], float]
# _AccuracyFn = Callable[[t.Tensor, t.Tensor], float]
# #_ModelApplyFn = Callable[..., chex.Array]
# _ModelApplyFn = Callable[..., t.Tensor]
# _MAX_RNGS_RESERVE = 50000

device = 'cuda' if t.cuda.is_available() else 'cpu'

@dataclasses.dataclass
class ClassicTrainingParams:
  """Parameters needed to train classical architectures."""
  
  # model_init_seed: int  # Used to initialize model parameters.
  training_steps: int
  # post_training_steps is used to train the model for more steps after the initial training
  #   this is used to finetune a model, for example switching from parallel to staircase
  second_training_steps: int
  log_frequency: int

  task: task_lib.GeneralizationTask
  length_curriculum: curriculum_lib.Curriculum
  batch_size: int

  weight_decay: float
  architecture: str

  single_output: bool

  model: nn.Module
  loss_fn: Callable[[t.Tensor, t.Tensor], tuple[float, _LossMetrics]]
  learning_rate: float
  test_model: Optional[nn.Module] = None
  max_grad_norm: float = 1.
  is_autoregressive: bool = False
  include_eos: bool = True

  tboard_logdir: str = None
  use_wandb: bool = False

  compute_full_range_test: bool = False
  range_test_total_batch_size: int = 512
  range_test_sub_batch_size: int = 64
  max_range_test_length: int = 100
  validation_set: List[task_lib.Batch] = None
  validate_steps: int = 1000
  validation_criteria: str = 'accuracy'

  accuracy_fn: Optional[Callable[[t.Tensor, t.Tensor],
                                 t.Tensor]] = None
  
  profile_model: bool = False

  

class TrainingWorker:
  """Training worker."""

  def __init__(self,
               training_params: ClassicTrainingParams,
               use_tqdm: bool = False,
               is_autoregressive: bool = False,
               computation_steps_mult: int = 0,
               ):
    """Initializes the worker.

    Args:
      training_params: The training parameters.
      use_tqdm: Whether to add a progress bar to stdout.
    """
    self._training_params = training_params
    self._use_tqdm = use_tqdm
    self._is_autoregressive = is_autoregressive
    self._computation_steps_mult = computation_steps_mult

    if training_params.validation_criteria == 'accuracy':
      training_params.best_validation_metric = 0
    elif training_params.validation_criteria == 'loss':
      training_params.best_validation_metric = sys.maxsize
      raise ValueError(f'validation_criteria {training_params.validation_criteria} is not supported yet')
    else:
      raise ValueError(f'validation_criteria {training_params.validation_criteria} is not supported yet')

    

    if training_params.tboard_logdir is not None:
      self.writer = SummaryWriter(log_dir=training_params.tboard_logdir)

  def run(
      self,
      optimizer: t.optim.Optimizer
  ) -> tuple[
      list[Mapping[str, Any]], Optional[list[Mapping[str, Any]]], t.Tensor
    ]:
    """Trains the model with the provided config.

    Returns:
      Results (various training and validation metrics), module parameters
      and router parameters.
    """
    training_params = self._training_params

    # What results?
    results = []

    model = training_params.model
    
    task = training_params.task
    length_curriculum = training_params.length_curriculum

    # architecture = training_params.architecture
    loss_fn = training_params.loss_fn
    accuracy_fn=training_params.accuracy_fn

    # TODO: Find a nicer way to go around this
    # Sequence padding function
    if self._is_autoregressive:
        raise ValueError("Autoregressive mode is not supported at the moment. Date: 24.01.2024.")
    else:
      pad_sequence = utils.pad_sequence_with_empty_targets(
        generalization_task=task,
        computation_steps_mult=self._computation_steps_mult,
        include_eos=training_params.include_eos
      )


    # best_model = model
    # best_accuracy = -1

    @t.inference_mode()
    def validate():
      model.eval()
      validation_losses = []
      validation_accuracies = []
      for length, validation_batch in enumerate(training_params.validation_set):
        output_length = validation_batch['output_length']
        validation_batch_input = validation_batch['input'].to(device)
        validation_batch_output = validation_batch['output'].to(device)

        output = model(validation_batch_input)

        reg_loss = None
        if type(output) is dict:
          reg_loss = output['reg_loss']
          output = output['output']

        if not training_params.single_output:
          output = output[:, -output_length:]

        validation_loss, _ = loss_fn(output, validation_batch_output)
        if accuracy_fn is not None:
          validation_accuracy = accuracy_fn(output, validation_batch_output)
        else:
          validation_accuracy = None
        validation_losses.append(float(t.mean(validation_loss)))
        validation_accuracies.append(float(t.mean(validation_accuracy)))
      return np.mean(validation_losses), np.mean(validation_accuracies)
    
    steps = range(training_params.training_steps + 1 + training_params.second_training_steps)
    if self._use_tqdm:
      steps = tqdm.tqdm(steps)

    if training_params.profile_model:
      start = t.cuda.Event(enable_timing=True)
      start.record()
    for step in steps:
      if (
        step == training_params.training_steps 
        and isinstance(model, my_staircase.MyStaircaseTransformer)
        and training_params.second_training_steps > 0
      ):
        model.mode = my_staircase.ForwardMode.STAIRCASE
        print('Switching the model to staircase mode')
      model.train()
      # Randomness handled by either python.random or numpy.
      length = length_curriculum.sample_sequence_length(step)

      output_length = task.output_length(length)
      # Randomness handled by either jax, python.random or numpy.
      #train_batch = task.sample_batch(
      #    next(rng_seq), length=length, batch_size=training_params.batch_size)
      train_batch = task.sample_batch(
          length=length, batch_size=training_params.batch_size)
      # Translation of code from "example.py" and "utils.py"
      if self._is_autoregressive:
        raise ValueError("Autoregressive mode is not supported at the moment. Date: 24.01.2024.")
      else:
        train_batch['input'] = pad_sequence(train_batch['input'])

      train_batch_input = train_batch['input'].to(device)
      train_batch_output = train_batch['output'].to(device)

      optimizer.zero_grad(set_to_none=True)
      if self._is_autoregressive:
        raise ValueError("Autoregressive mode is not supported at the moment. Date: 24.01.2024.")
      else:
        #start_forward = t.cuda.Event(enable_timing=True)
        #end_forward = t.cuda.Event(enable_timing=True)
        #start_forward.record()
        output = model(train_batch_input)

      #print(output.shape)
      #print(train_batch['output'].shape)
      ## import sys
      # sys.exit()

      reg_loss = None
      if type(output) is dict:
        reg_loss = output['reg_loss']
        output = output['output']

      if not training_params.single_output:
        output = output[:, -output_length:]

      train_loss, train_metrics = loss_fn(output, train_batch_output)
      #end_forward.record()
      #t.cuda.synchronize()
      #forward_t = start_forward.elapsed_time(end_forward)
      if accuracy_fn is not None:
        train_accuracy = accuracy_fn(output, train_batch_output)
        # if train_accuracy > best_accuracy:
        #   best_model = model
      else:
        train_accuracy = None

      #start_backward = t.cuda.Event(enable_timing=True)
      #end_backward = t.cuda.Event(enable_timing=True)
      #start_backward.record() 
      if reg_loss:
        train_loss = train_loss + reg_loss
      train_loss.backward()
      #end_backward.record()
      #t.cuda.synchronize()
      #backward_t = start_backward.elapsed_time(end_backward)
      #print(f'Step {step:3}: forward {forward_t:.0f}ms\tbackward{backward_t:.0f}ms')
      nn.utils.clip_grad_norm_(model.parameters(), training_params.max_grad_norm)
      optimizer.step()

      if step % training_params.validate_steps == 0 and training_params.validation_set is not None:
        validation_loss, validation_accuracy = validate()
        #print(validation_loss)
        #print(validation_accuracy)
      else:
        validation_accuracy = validation_loss = None

      log_freq = training_params.log_frequency
      if (log_freq > 0) and (step % log_freq == 0):
        log_data = {
            "step": step,
            "train_loss": float(train_loss),
        }
        if training_params.accuracy_fn is not None:
          log_data["train_accuracy"] = float(train_accuracy)
        for key, value in train_metrics.items():
          log_data[".".join(["train_metrics", key])] = np.array(value)
        results.append(log_data)

        if training_params.profile_model:
          end = t.cuda.Event(enable_timing=True)
          end.record()
          t.cuda.synchronize()
          elapsed_time_ms = start.elapsed_time(end)
          elapsed_time_per_step_ms = elapsed_time_ms / log_freq
          print(f'Elapsed time per step: {elapsed_time_per_step_ms}ms')
          start = t.cuda.Event(enable_timing=True)
          start.record()

        if training_params.use_wandb:
          wandb.log(
            {
              'train': {
                'loss': float(train_loss),
                'accuracy': float(train_accuracy),
              },
            },
            step=step
          )

          if validation_loss:
            wandb.log(
              {
                'validation2': {
                  'loss': validation_loss,
                  'accuracy': validation_accuracy
                }
              },
              step=step
            )

        if self._training_params.tboard_logdir is not None:
          self.writer.add_scalar("Loss/train", float(train_loss), step)
          self.writer.add_scalar("Accuracy/train", float(train_accuracy), step)

      # We need to access this private attribute since the default reserve size
      # can not be edited yet.
      #if not rng_seq._subkeys:  # pylint: disable=protected-access
      #  rng_seq.reserve(rngs_reserve)

    # print(results)


    eval_results = None
    if training_params.compute_full_range_test:
      eval_params = range_evaluation.EvaluationParams(
          # model=training_params.test_model or model,
          model=model,
          accuracy_fn=training_params.accuracy_fn,
          sample_batch=task.sample_batch,
          max_test_length=training_params.max_range_test_length,
          total_batch_size=training_params.range_test_total_batch_size,
          sub_batch_size=training_params.range_test_sub_batch_size,
          is_autoregressive=training_params.is_autoregressive,
          task=task,
          computation_steps_mult=self._computation_steps_mult,
          single_output=training_params.single_output,
          use_wandb=training_params.use_wandb,
          include_eos=training_params.include_eos
      )

      writer = self.writer if hasattr(self, 'writer') else None
      eval_results = range_evaluation.range_evaluation(
          eval_params, use_tqdm=False, tboard_writer=writer)

    if self._training_params.tboard_logdir is not None:
      self.writer.flush()
      self.writer.close()
    #return results, eval_results, params
    return results, eval_results
