import copy
import os
import random

# Ensure environment variables are set before TensorFlow / framework initialization
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import keras
import numpy
import pytest
import tensorflow as tf

from tests.testing_utils import (
    PROBLEM_DEFAULTS,
    _get_classification_data,
    _get_regression_data,
)

DEFAULT_SEED = 0


@pytest.fixture(autouse=True)
def set_random_seed():
    """Establish deterministic test environment across Python, NumPy, TensorFlow, and Keras."""
    seed_value = DEFAULT_SEED

    # 1. Use modern Keras API to set Python, NumPy, and framework backend seeds
    keras.utils.set_random_seed(seed_value)

    # 2. Explicitly set random, numpy, and tensorflow seeds for comprehensive coverage
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    tf.random.set_seed(seed_value)


@pytest.fixture
def classification_data():
    """Fixture providing deterministic Iris classification train/test split (90/10)."""
    return _get_classification_data()


@pytest.fixture
def regression_data():
    """Fixture providing deterministic synthetic regression train/test split."""
    return _get_regression_data()


@pytest.fixture
def problem_defaults():
    """Fixture providing standard problem type default configurations."""
    return copy.deepcopy(PROBLEM_DEFAULTS)