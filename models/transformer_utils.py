#
# Copyright 2024- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

from enum import Enum

class TransformerNormalization(Enum):
    prenorm = 0
    postnorm = 1
    perinorm = 2

class InitType(Enum):
    pytorch = 0
    moeut = 1
