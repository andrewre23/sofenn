"""Utils to be used across tests."""
import copy
from functools import lru_cache
from pathlib import Path

import keras
import numpy
import pandas
from keras import activations
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, Accuracy
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
            'metrics': [CategoricalAccuracy]
        }
    },
    # 'logistic_regression': {
    #     'samples': 50,
    #     'features': 4,
    #     'neurons': 3,
    #     'num_classes': 1,
    #     'activation': activations.sigmoid,
    #     'compile': {
    #         'loss': BinaryCrossentropy,
    #         'optimizer': Adam,
    #         'metrics': [Accuracy]
    #     }
    # },
    'classification': {
        'samples': 50,
        'features': 4,
        'neurons': 5,
        'num_classes': 3,
        'activation': activations.softmax,
        'compile': {
            'loss': MeanSquaredError,
            'optimizer': RMSprop,
            'metrics': [Accuracy]
        }
    }

}
PROBLEM_TYPES = [
    {'testcase_name': problem_type, 'problem_type': problem_type} for problem_type in PROBLEM_DEFAULTS.keys()
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
    elif 'regression' in problem_type:
        return _get_regression_data(problem_type)

@lru_cache(maxsize=None)
def _get_classification_data():
    """Load iris data for classification."""
    features = pandas.read_csv(DATA_DIR / 'iris/features.csv')
    target = pandas.read_csv(DATA_DIR / 'iris/target.csv')
    return train_test_split(features.values, target.values, test_size=0.1, random_state=23)

@lru_cache(maxsize=None)
def _get_regression_data(problem_type):
    """Generate data for a regression based on input shape provided."""
    defaults = PROBLEM_DEFAULTS[problem_type]

    x = numpy.random.random((defaults['samples'], defaults['features']))
    noise = numpy.random.normal(0, .5, (defaults['samples'], defaults['num_classes']))
    W = numpy.random.randint(0, 10, (defaults['features'], defaults['num_classes']))
    y = numpy.dot(x, W) + noise

    def sigmoid(inputs):
        return 1 / (1 + numpy.exp(-inputs))
    y = sigmoid(y) if problem_type == 'logistic_regression' else y

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
    elif 'regression' in problem_type:
        model = f'models/{problem_type}{deep}.keras'
    return keras.saving.load_model(DATA_DIR / model, custom_objects={'FuzzyNetwork': FuzzyNetwork})
