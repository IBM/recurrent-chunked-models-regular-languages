#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Example script to train and evaluate a network."""

import torch as t
import os

import sys
# Put this package on my path
sys.path.append(f'{os.path.dirname(os.getcwd())}')

import pickle

from absl import app
from absl import flags
import numpy as np
import recurrent_chunked_models_regular_languages.wandb_wrapper as wandb
import random

from recurrent_chunked_models_regular_languages.experiments import constants
from recurrent_chunked_models_regular_languages.experiments import curriculum as curriculum_lib
from recurrent_chunked_models_regular_languages.experiments import training
from recurrent_chunked_models_regular_languages.models.transformer_utils import TransformerNormalization, InitType
from recurrent_chunked_models_regular_languages.experiments import utils
from recurrent_chunked_models_regular_languages.models.my_staircase import PRETRAINED_CHECKPOINTS

_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    default=128,
    help='Training batch size.',
    lower_bound=1,
)

_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length',
    default=40,
    help='Maximum training sequence length.',
    lower_bound=1,
)

_TASK = flags.DEFINE_string(
    'task',
    default='parity_check',
    help='Length generalization task (see `constants.py` for other tasks).',
)

_ARCHITECTURE = flags.DEFINE_string(
    'architecture',
    default='rnn',
    help='Model architecture (see `constants.py` for other architectures).',
)

_IS_AUTOREGRESSIVE = flags.DEFINE_boolean(
    'is_autoregressive',
    default=False,
    help='Whether to use autoregressive sampling or not.',
)

_COMPUTATION_STEPS_MULT = flags.DEFINE_integer(
    'computation_steps_mult',
    default=0,
    help=(
        'The amount of computation tokens to append to the input tape (defined'
        ' as a multiple of the input length)'
    ),
    lower_bound=0,
)

_SEED = flags.DEFINE_integer(
  'seed',
  default=None,
  help='random seed.'
)

_LEARNING_RATE = flags.DEFINE_float(
  'lr',
  default=1e-5,
  help='learning rate'
)

_WEIGHT_DECAY = flags.DEFINE_float(
  'wd',
  default=0.0,
  help='weight decay.'
)

_ADAM_BETA1_INV = flags.DEFINE_float(
  'adam_beta1_inv',
  default=.1,
  help='adam_beta1=1-adam_beta1_inv'
)

_TRAIN_STEPS = flags.DEFINE_integer(
  'train_steps',
  default=0,
  help='number of training steps.'
)

_TEST_LENGTH = flags.DEFINE_integer(
  'test_length',
  default=500,
  help='maximum test sequence length.'
)

_NUM_LAYERS = flags.DEFINE_integer(
  'num_layers',
  default=None,
  help='number of layers'
)

_EMBED_SIZE = flags.DEFINE_integer(
  'embed_size',
  default=32,
  help='embedding size'
)

# TODO: I'm not sure how to think about the EOS token with regards to non-regular languages
_INCLUDE_EOS = flags.DEFINE_integer(
  'include_eos',
  default = 1,
  help = 'Should the EOS token be appened to the input X?',
  lower_bound=0,
  upper_bound=1,
)

_STATE_SIZE = flags.DEFINE_integer(
  'state_size',
  default=256,
  help='linear models state size'
)

_NUM_AS = flags.DEFINE_integer(
  'num_as',
  default=2,
  help='number of A matrices in the selective linear RNN'
)

_A_INIT_VARIANCE = flags.DEFINE_float(
  'a_var',
  default=0.1,
  help='variance of state transition matrix A at initialization'
)

_TRAIN_H0 = flags.DEFINE_boolean(
  'train_h0',
  default = True,
  help='set initial state of recurrent models as trainable (supported in lsrnn)'
)

_NO_NORM = flags.DEFINE_boolean(
  'no_norm',
  default = False,
  help = 'disable normalization (supported in lsrnn)'
)

_NO_RESIDUAL = flags.DEFINE_boolean(
  'no_residual',
  default=False,
  help = 'disable residual (supported in lsrnn)'
)

_SEL_TEMP = flags.DEFINE_boolean(
  'select_temp',
  default=False,
  help = 'scale the selection softmax using a trainable temperature parameter (in lsrnn)'
)

_SEL_TEMP_PARAM = flags.DEFINE_string(
  'select_temp_param',
  default='exp',
  help = 'parametrization of the selection softmax temperature (in lsrnn)'
)

_OUT_DIR = flags.DEFINE_string(
  'out_dir',
  default='out',
  help = 'Where should the output be written to?'
)

_NUM_HEADS = flags.DEFINE_integer(
  'num_heads',
  default='4',
  help = 'How many heads should the transformer models have?'
)

_DROPOUT = flags.DEFINE_float(
  'dropout',
  default='0.0',
  help = 'How much dropout to apply?'
)

_GROUP_SIZE = flags.DEFINE_integer(
  'group_size',
  default='1',
  help = 'How many layers in a universal transformer group?'
)

_NUM_RECURRENT_CALLS = flags.DEFINE_integer(
  'num_recurrent_calls',
  default=None,
  help = 'How many recurrent calls should a universal transformer have?'
)

_IS_CAUSAL = flags.DEFINE_integer(
  'is_causal',
  default=0,
  lower_bound=0,
  upper_bound=1,
  help='Whether to apply a causal mask to attention. Value must be {0, 1}.'
)

_LAYER_NORM_TYPE = flags.DEFINE_enum(
  'norm_type',
  default=TransformerNormalization.prenorm.name,
  enum_values=list(TransformerNormalization.__members__),
  help='Where is the transformer layer norm placed? [prenorm, postnorm, perinorm]'
)

_INIT_TYPE = flags.DEFINE_enum(
  'init_type',
  default=InitType.pytorch.name,
  enum_values=list(InitType.__members__),
  help='What is the initalization type? [pytorch, moeut]'
)

_LOAD_PRETRAIN = flags.DEFINE_boolean(
  'load_pretrain',
  default=False,
  help=(
    'Whether to load weights from a pretrained model. If True, a random checkpoint is selected from '
    'my_staircase.PRETRAINED_CHECKPOINTS'
  )
)

###############################
# MoEUT Flags
###############################

_FF_N_EXPERTS = flags.DEFINE_integer(
  'ff_n_experts',
  default=None,
  lower_bound=1,
  help='The number of feedforward experts'
)

_ATT_N_EXPERTS = flags.DEFINE_integer(
  'att_n_experts',
  default=None,
  lower_bound=1,
  help='The number of attention experts'
)

_FF_K = flags.DEFINE_integer(
  'ff_k',
  default=None,
  lower_bound=1,
  help='The number of feedforward experts to keep'
)

_ATT_K = flags.DEFINE_integer(
  'att_k',
  default=None,
  lower_bound=1,
  help='The number of attention experts to keep'
)

_FF_EXPERT_SIZE = flags.DEFINE_integer(
  'ff_expert_size',
  default=None,
  lower_bound=1,
  help='The width of each expert'
)

_POSITIONAL = flags.DEFINE_enum(
  'positional',
  default=None,
  enum_values=['nope', 'rope'],
  help='The type of positional encoding [nope, rope]'
)

###############################
# Staircase Flags
###############################
_ATTN_LIM = flags.DEFINE_integer(
  'attn_lim',
  default=500,
  help='The max context length of attention. This is used for relative positional embeddings in Staircase'
)

_FIX_STAIRCASE_SIZE_FORWARD = flags.DEFINE_integer(
  'fix_staircase_size_forward',
  default=-1,
  help='max number of fresh tokens considered'
)

_STAIRCASE_CACHE_STEPS = flags.DEFINE_integer(
  'staircase_cache_steps',
  default=0,
  help=(
      'The number of processing steps to cache finished chunks. Set to -1 to cache until all tokens are processed ' 
      '(ie the global cached staircase from Figure 1)'
  )
)

_STAIRCASE_SIZE = flags.DEFINE_integer(
  'staircase_size',
  default=-1,
  help='number of tokens in each transformer forward'
)

_MEM_SZ = flags.DEFINE_integer(
  'mem_sz',
  default=-1,
  help='I\'m not really sure how this parameter relates to staircase_size, investigate'
)

_SECOND_TRAINING_STEPS = flags.DEFINE_integer(
  'second_training_steps',
  default=0,
  help='The number of steps to finetune the model in staircase mode'
)

_DIAGONAL = flags.DEFINE_boolean(
  'diagonal',
  default=False,
  help='If True, staircase attention also uses the diagonal (ie normal parallel attention).'
)

###############################
# Block Recurrent Transformer Flags
###############################
_CHUNK_SIZE = flags.DEFINE_integer(
  'chunk_size',
  default=None,
  help='The chunk size'
)

_NUM_STATE_VECTORS = flags.DEFINE_integer(
  'num_state_vectors',
  default=None,
  help='The number of state vectors'
)

_RECURRENT_LAYER_INDEX = flags.DEFINE_integer(
  'recurrent_layer_index',
  default=None,
  help='What layer should be the recurrent one?'
)

###############################
# Parallel Parity Check Flags
###############################

_NUM_PARITY_AUTOMATA = flags.DEFINE_integer(
  'num_parity_automata',
  default=2,
  help='number of parallel parity check automata for the parallel_parity_check task'
)

_PARITY_EXPLICIT_DICTIONARIES = flags.DEFINE_boolean(
  'explicit_tokens_parity',
  default=True,
  help='use different dictionaries for different parity automata?'
)

###############################
# Other Flags
###############################

_COMPILE = flags.DEFINE_boolean(
  'compile',
  default=False,
  help='Whether to run torch.compile()'
)

# W&B Flags
_USE_WANDB = flags.DEFINE_boolean(
  'use_wandb',
  default=False,
  help='Whether to turn on W&B'
)

_DISPLAY_NAME = flags.DEFINE_string(
  'display_name',
  default='',
  help = 'The display name of this experiment to log to W&B'
)

_INCREASE_CURRICULUM = flags.DEFINE_boolean(
    'increase_curriculum',
    default=False,
    help='If True, the training length increases by 1 each step. This is useful for debugging.',
)

_EXPERIMENT_DESCRIPTION = flags.DEFINE_string(
  'expr_desc',
  default='',
  help = 'The description of this experiment to log to W&B'
)

_WANDB_GROUP = flags.DEFINE_string(
  'wandb_group',
  default=None,
  help = 'The wandb group'
)

_WANDB_PROJECT = flags.DEFINE_string(
  'wandb_project',
  default='chomsky',
  help = 'The wandb project'
)

_VALIDATE = flags.DEFINE_boolean(
    'validate',
    default=False,
    help=(
      'If True, a validation set is generated and used for early stopping and checkpoint selection. Otherwise, the '
      'model is trained for a fixed number of steps and uses the last checkpoint at test time.'
    )
)

_VALIDATION_LENGTH = flags.DEFINE_integer(
  'validation_length',
  default=None,
  help=(
    'Maximum validation sequence length. If set to None, uses `sequence_length`.'
  )
)

###############################
# Profiling Flags
###############################
_PROFILE_MODEL = flags.DEFINE_bool(
  'profile_model',
  default=False,
  help=(
    'Whether to profile the speed of each batch. WARNING: this will slow down processing a lot, only turn it on for '
    'profiling experiments.'
  )
)


_ARCHITECTURE_PARAMS = {}

# TODO: add a blocker for uncommitted changes, this effectively saves the code based on the git SHA
def init_wandb(args) -> wandb.sdk.wandb_run.Run:
  config = dict(args)
  
  # Get environment variables
  config['Job ID'] = os.environ.get('LSB_JOBID')
  config['Description'] = _EXPERIMENT_DESCRIPTION.value
  config['Shell cmd'] = os.environ.get('SHELL_CMD')
  # Wrap the values that contain whitespace in single quotes
  arguments = []
  for argument in sys.argv:
      if ' ' in argument:
          # Arguments and values can come in the form --arg_name='arg_value' or --arg_name 'arg_value'.
          # Support both here
          if argument[:2] == '--' and '=' in argument:
            arg_name, arg_value = argument.split('=')
            # NOTE: the double quotes on the outside here are important, don't change it
            arguments.append(f"{arg_name}='{arg_value}'")
          else:
            # NOTE: the double quotes on the outside here are important, don't change it
            arguments.append(f"'{argument}'")
      else:
        arguments.append(argument)
  config['Python cmd'] = f'{sys.executable} {" ".join(arguments)}'

  if not _DISPLAY_NAME.value:
    # Use the job ID if there is no display name, if there is no job id use None and wandb will generate one.
    display_name = os.environ.get('LSB_JOBID', None)
  else:
    display_name = _DISPLAY_NAME.value

  run = wandb.init(
    project=_WANDB_PROJECT.value,
    name=display_name,
    config=config,
    group=_WANDB_GROUP.value
  )
  return run


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  t.manual_seed(seed)


def main(unused_argv) -> None:
  # This shouldn't be necessary, but there's something wrong with VSCode's python debugger skipping breakpoints at the
  # moment
  if os.environ.get('DEBUG', 0):
    breakpoint()
  args = flags.FLAGS

  if _SEED.value is None:
    args.seed = random.randint(0, 2**32 - 1)

  if _VALIDATION_LENGTH.value is None:
    args.validation_length = args.sequence_length
  set_seed(_SEED.value)

  print(f'Cuda is available? {t.cuda.is_available()}')
  # Create the task.
  if _INCREASE_CURRICULUM.value:
    print('Using increase curriculum')
    curriculum = curriculum_lib.RegularIncreaseCurriculum(
      initial_sequence_length=1,
      increase_frequency=1,
      increase_amount=1,
      sample_all_length=False
    )
  else:
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(1, _SEQUENCE_LENGTH.value + 1))
    )

  if _TASK.value == 'parallel_parity_check':
    task = constants.TASK_BUILDERS[_TASK.value](
      n = _NUM_PARITY_AUTOMATA.value,
      explicit_tokens = _PARITY_EXPLICIT_DICTIONARIES.value
    )
  else:
    task = constants.TASK_BUILDERS[_TASK.value]()
  print(f"Task: {_TASK.value}")

  # Create the model.
  single_output = task.output_length(10) == 1

  # Stash all of the arguments in _ARCHITECTURE_PARAMS
  for arg in args:
    _ARCHITECTURE_PARAMS[str(arg)] = args[str(arg)]._value
  _ARCHITECTURE_PARAMS['norm_type'] = TransformerNormalization[_ARCHITECTURE_PARAMS['norm_type']]
  _ARCHITECTURE_PARAMS['init_type'] = InitType[_ARCHITECTURE_PARAMS['init_type']]

  _ARCHITECTURE_PARAMS['num_layers'] = _NUM_LAYERS.value
  _ARCHITECTURE_PARAMS['embed_size'] = _EMBED_SIZE.value
  _ARCHITECTURE_PARAMS['hidden_size'] = _EMBED_SIZE.value
  _ARCHITECTURE_PARAMS['state_size'] = _STATE_SIZE.value
  _ARCHITECTURE_PARAMS['max_seq_len'] = max(_SEQUENCE_LENGTH.value, _TEST_LENGTH.value)
  _ARCHITECTURE_PARAMS['num_A'] = _NUM_AS.value
  _ARCHITECTURE_PARAMS['A_init_variance'] = _A_INIT_VARIANCE.value
  _ARCHITECTURE_PARAMS['no_norm'] = _NO_NORM.value
  _ARCHITECTURE_PARAMS['train_h0'] = _TRAIN_H0.value
  _ARCHITECTURE_PARAMS['no_residual'] = _NO_RESIDUAL.value
  _ARCHITECTURE_PARAMS['select_temp'] = _SEL_TEMP.value
  _ARCHITECTURE_PARAMS['select_temp_param'] = _SEL_TEMP_PARAM.value
  _ARCHITECTURE_PARAMS['d_model'] =  _ARCHITECTURE_PARAMS['embed_size']
  assert _ARCHITECTURE_PARAMS['d_model'] // _ARCHITECTURE_PARAMS['num_heads']
  _ARCHITECTURE_PARAMS['head_dim'] = _ARCHITECTURE_PARAMS['d_model'] // _ARCHITECTURE_PARAMS['num_heads']

  if _USE_WANDB.value:
    run = init_wandb(_ARCHITECTURE_PARAMS)
  else:
    run = None
  
  
#   print(f'single_output is {single_output}')

  extra_dims_onehot = 1 + int(_COMPUTATION_STEPS_MULT.value > 0)
  final_input_size = task.input_size + extra_dims_onehot
  
  model = constants.MODEL_BUILDERS[_ARCHITECTURE.value](
      output_size=task.output_size,
      return_all_outputs=not single_output,
      input_size=final_input_size,
      **_ARCHITECTURE_PARAMS,
  )

  if _COMPILE.value:
    print('Compiling the model')
    model = t.compile(model)

  model = model.to('cuda' if t.cuda.is_available() else 'cpu')

  if run:
    parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.config.update({
      'parameters': f'{parameters:,}',
      'trainable_parameters': f'{trainable_parameters:,}'
    })
  
  # print("Model configuration: \n")
  print(model)
  model_num_parameters = sum(p.numel() for p in model.parameters())
  print(f"Total number of parameters: {model_num_parameters}")
  print(f'Hyperparameters: batchsize={_BATCH_SIZE.value}, seed={_SEED.value}, lr={_LEARNING_RATE.value}, wd={_WEIGHT_DECAY.value}\n')


  # Result paths
  if 'selective' in _ARCHITECTURE.value:
    modelname = f"{_ARCHITECTURE.value}_{_TASK.value}_params_{model_num_parameters}_layers_{_ARCHITECTURE_PARAMS['num_layers']}_embed_{_ARCHITECTURE_PARAMS['embed_size']}_state_{_ARCHITECTURE_PARAMS['state_size']}_numA_{_ARCHITECTURE_PARAMS['num_A']}_Avar_{_ARCHITECTURE_PARAMS['A_init_variance']}_LR_{_LEARNING_RATE.value}_WD_{_WEIGHT_DECAY.value}_trainstep_{_TRAIN_STEPS.value}_trainlen_{_SEQUENCE_LENGTH.value}_SEED_{_SEED.value}"
  elif _ARCHITECTURE.value == 'lsrnn':
    modelname = f"{_ARCHITECTURE.value}_{_TASK.value}_params_{model_num_parameters}_layers_{_ARCHITECTURE_PARAMS['num_layers']}_embed_{_ARCHITECTURE_PARAMS['embed_size']}_state_{_ARCHITECTURE_PARAMS['state_size']}_numA_{_ARCHITECTURE_PARAMS['num_A']}_Avar_{_ARCHITECTURE_PARAMS['A_init_variance']}_trainh0_{_TRAIN_H0.value}_nonorm_{_NO_NORM.value}_nores_{_NO_RESIDUAL.value}_seltemp_{_SEL_TEMP.value}_stpar_{_SEL_TEMP_PARAM.value}_LR_{_LEARNING_RATE.value}_WD_{_WEIGHT_DECAY.value}_trainstep_{_TRAIN_STEPS.value}_trainlen_{_SEQUENCE_LENGTH.value}_SEED_{_SEED.value}"
  else:
    modelname = f"{_ARCHITECTURE.value}_{_TASK.value}_params_{model_num_parameters}_layers_{_ARCHITECTURE_PARAMS['num_layers']}_embed_{_ARCHITECTURE_PARAMS['embed_size']}_state_{_ARCHITECTURE_PARAMS['state_size']}_LR_{_LEARNING_RATE.value}_WD_{_WEIGHT_DECAY.value}_trainstep_{_TRAIN_STEPS.value}_trainlen_{_SEQUENCE_LENGTH.value}_SEED_{_SEED.value}"
  savedir = f"{_OUT_DIR.value}/{_ARCHITECTURE.value}/{_TASK.value}/"
  modelpath = os.path.join(savedir, f'{modelname}.pt')

  if os.path.exists(modelpath):
    print(f"selected model is already trained: {modelpath}")
    return

  # Create the loss and accuracy based on the pointwise ones.
  def loss_fn(output, target):
    loss = t.mean(t.sum(task.pointwise_loss_fn(output, target), axis=-1))
    # print(f"Loss: {loss}")
    return loss, {}

  def accuracy_fn(output, target):
    mask = task.accuracy_mask(target)
    return t.sum(mask * task.accuracy_fn(output, target)) / t.sum(mask)
  
  if _VALIDATE.value:
    pad_sequence = utils.pad_sequence_with_empty_targets(
        generalization_task=task,
        computation_steps_mult=_COMPUTATION_STEPS_MULT.value,
        include_eos=args.include_eos,
    )
    validation_set = task.generate_validation_set(
      batch_size=_BATCH_SIZE.value,
      max_length=args.validation_length,
      pad_sequence=pad_sequence
    )
  else:
    validation_set = None

  # Create the final training parameters.
  training_params = training.ClassicTrainingParams(
      training_steps=_TRAIN_STEPS.value,
      log_frequency=100,
      length_curriculum=curriculum,
      batch_size=_BATCH_SIZE.value,
      task=task,
      model=model,
      loss_fn=loss_fn,
      learning_rate=_LEARNING_RATE.value,
      weight_decay=_WEIGHT_DECAY.value,
      accuracy_fn=accuracy_fn,
      compute_full_range_test=True,
      max_range_test_length=_TEST_LENGTH.value,
      range_test_total_batch_size=512,
      range_test_sub_batch_size=64,
      is_autoregressive=_IS_AUTOREGRESSIVE.value,
      architecture=_ARCHITECTURE.value,
      single_output=single_output,
      tboard_logdir=os.path.join(savedir, modelname),
      use_wandb=_USE_WANDB.value,
      validation_set=validation_set,
      second_training_steps=_SECOND_TRAINING_STEPS.value,
      include_eos=args.include_eos,
      profile_model=_PROFILE_MODEL.value
  )

  training_worker = training.TrainingWorker(training_params, use_tqdm=True,
                                            is_autoregressive=_IS_AUTOREGRESSIVE.value,
                                            computation_steps_mult=_COMPUTATION_STEPS_MULT.value)
  optimizer = t.optim.AdamW(
    model.parameters(),
    lr=training_params.learning_rate,
    betas=(1-args.adam_beta1_inv, .999),
    weight_decay=training_params.weight_decay,
  )

  if args.load_pretrain:
    pretrained_checkpoint_path = random.choice(PRETRAINED_CHECKPOINTS)
    print(f'Using pretrained checkpoint {pretrained_checkpoint_path}')
    checkpoint = t.load(pretrained_checkpoint_path, map_location='cuda' if t.cuda.is_available() else 'cpu')
    # I think that when finetuning, the optimizer state should not be loaded.
    # optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])

  train_results, eval_results = training_worker.run(optimizer)

  print(train_results)

  # Gather results and print final score.
  accuracies = [r['accuracy'] for r in eval_results]
  score = np.mean(accuracies[_SEQUENCE_LENGTH.value + 1 :])
  print(f'Network score: {score}')
  if _USE_WANDB.value:
    wandb.log({
      'score': score
    })
  
  os.makedirs(savedir, exist_ok=True)
  open(os.path.join(savedir, f"{modelname}_testlen_{_TEST_LENGTH.value}_accuracy_{score}"), 'w').close()

  checkpoint = {
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
  }
  t.save(checkpoint, modelpath)

  if _USE_WANDB.value:
    wandb.finish()

if __name__ == '__main__':
  app.run(main)
