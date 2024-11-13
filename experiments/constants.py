#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Constants for our length generalization experiments."""

from recurrent_chunked_models_regular_languages.experiments import curriculum as curriculum_lib

"""
Modules
"""
from recurrent_chunked_models_regular_languages.models import rnn
from recurrent_chunked_models_regular_languages.models import universal_transformer
from recurrent_chunked_models_regular_languages.models import my_staircase
try:
    from recurrent_chunked_models_regular_languages.models import block_recurrent_transformer
except:
    print("block_recurrent_transformer is not installed")
    # Create a generic object to wrap block_recurrent_transformer so that we don't cause an error when creating 
    # MODEL_BUILDERS
    class Object(object):
        pass
    block_recurrent_transformer = Object()
    block_recurrent_transformer.BlockRecurrentTransformerWrapper = Object()
from recurrent_chunked_models_regular_languages.models import transformer

"""
Tasks
"""
from recurrent_chunked_models_regular_languages.tasks.regular import cycle_navigation
from recurrent_chunked_models_regular_languages.tasks.regular import even_pairs
from recurrent_chunked_models_regular_languages.tasks.regular import modular_arithmetic
from recurrent_chunked_models_regular_languages.tasks.regular import parity_check


MODEL_BUILDERS = {
    'rnn':
        rnn.ElmanRNN,
    'my_staircase':
        my_staircase.MyStaircaseTransformer,
    'brt':
        block_recurrent_transformer.BlockRecurrentTransformerWrapper,
    'transformer':
        transformer.Transformer,
}

CURRICULUM_BUILDERS = {
    'fixed': curriculum_lib.FixedCurriculum,
    'regular_increase': curriculum_lib.RegularIncreaseCurriculum,
    'reverse_exponential': curriculum_lib.ReverseExponentialCurriculum,
    'uniform': curriculum_lib.UniformCurriculum,
}

TASK_BUILDERS = {
    'modular_arithmetic':
        modular_arithmetic.ModularArithmetic,
    'parity_check':
        parity_check.ParityCheck,
    'even_pairs':
        even_pairs.EvenPairs,
    'cycle_navigation':
        cycle_navigation.CycleNavigation,
}

TASK_LEVELS = {
    'modular_arithmetic': 'regular',
    'parity_check': 'regular',
    'even_pairs': 'regular',
    'cycle_navigation': 'regular',
}
