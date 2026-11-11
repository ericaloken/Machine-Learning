# test/test_optimizers.py

"""
Tests for optimizers.py
"""

import numpy as np
from jax.tree_util import tree_map


from optimizers import GradientDescent, Momentum, AdaGrad, RMSprop, Adam
