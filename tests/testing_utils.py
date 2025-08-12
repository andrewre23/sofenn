"""Utils to be used across tests."""
import copy
from functools import lru_cache
from pathlib import Path

import keras
import numpy
import pandas
from keras import activations
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, MeanSquaredError
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split

from sofenn import FuzzyNetwork
from sofenn.losses import CustomLoss

DATA_DIR = Path(__file__).parent / 'data'

PROBLEM_DEFAULTS = {
    'regression': {
        'samples': 100,
        'features': 5,
        'neurons': 3,
        'num_classes': 1,
        'activation': activations.linear,
        'compile': {
            'loss': CustomLoss,
            'optimizer': Adam,
            'metrics': [MeanSquaredError]
        }
    },
    'classification': {
        'samples': 50,
        'features': 4,
        'neurons': 5,
        'num_classes': 3,
        'activation': activations.softmax,
        'compile': {
            'loss': CategoricalCrossentropy,
            'optimizer': RMSprop,
            'metrics': [CategoricalAccuracy]
        }
    }
# TODO: add logistic regression
}
PROBLEM_TYPES = [
    {'testcase_name': problem_type, 'problem_type': problem_type} for problem_type in PROBLEM_DEFAULTS.keys()
]

VALID_SHAPES = {
    '1D': (None,),
    '2D': (100, None),
    #    '3D': (50, 8, None),
}
SHAPES = [
    {'testcase_name': name, 'name': name, 'shape': shape} for name, shape in VALID_SHAPES.items()
]

def _init_params(problem_type='regression', **kwargs):
    """Generate default parameter dictionary for each problem type."""
    params = copy.deepcopy(PROBLEM_DEFAULTS.get(problem_type, {}))
    params.pop('compile')
    params['input_shape'] = (params.pop('samples', None), params.pop('features', None))
    for key, value in kwargs.items():
        params[key] = value
    return params

def _compile_params(problem_type='regression', **kwargs):
    """Generate default parameter dictionary for each problem type."""
    defaults = copy.deepcopy(PROBLEM_DEFAULTS[problem_type]['compile'])
    params = {key: [v() for v in val] if isinstance(val, list) else val() for key, val in defaults.items()}
    for key, value in kwargs.items():
        params[key] = value
    return params

@lru_cache(maxsize=None)
def _get_training_data(problem_type):
    if problem_type == 'classification':
        return _get_classification_data()
    elif problem_type == 'regression':
        return _get_regression_data()

@lru_cache(maxsize=None)
def _get_classification_data():
    """Load iris data for classification."""
    features = pandas.read_csv(DATA_DIR / 'iris/features.csv')
    target = pandas.read_csv(DATA_DIR / 'iris/target.csv')
    return train_test_split(features.values, target.values, test_size=0.1, random_state=23)

@lru_cache(maxsize=None)
def _get_regression_data():
    """Generate data for a regression based on input shape provided."""
    defaults = PROBLEM_DEFAULTS['regression']

    x_shape = (defaults['samples'], defaults['features'])
    x = numpy.random.random(x_shape)
    noise = numpy.random.normal(0, .5, (defaults['samples'], 1))
    W = numpy.random.randint(0, 10, (defaults['features'], defaults['num_classes']))
    y = numpy.dot(x, W) + noise

    train_samples = int(0.75 * len(x))
    X_train, X_test = x[:train_samples], x[train_samples:]
    y_train, y_test = y[:train_samples], y[train_samples:]
    return X_train, X_test, y_train, y_test

@lru_cache(maxsize=None)
def _load_saved_model(problem_type, deep=False):
    """Load pre-trained models."""
    deep = "-deep" if deep else ""
    if problem_type == 'classification':
        model =  f'models/iris_classification{deep}.keras'
    elif problem_type == 'regression':
        model = f'models/{problem_type}{deep}.keras'
    return keras.saving.load_model(DATA_DIR / model, custom_objects={'FuzzyNetwork': FuzzyNetwork})
