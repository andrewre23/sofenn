import copy
from typing import Union, Callable

import inspect
import keras.src.backend as k
from keras import activations
from keras.models import Model


def remove_nones(shape: tuple, value):
    """Remove none values from input shape."""
    return tuple([s if s is not None else value for s in shape]) if None in shape else shape

def replace_last_dim(shape: tuple, value: int):
    """Replace the last dimension of input shape."""
    new_shape = list(shape)
    new_shape[-1] = value
    return k.standardize_shape(tuple(new_shape))

def make_2d(shape: tuple):
    """Prepend 1 dimension for batch size 1 to input shape if input is 1D."""
    return (1,) + shape if len(shape) == 1 else shape

def is_valid_activation(activation: Union[str, Callable]):
    """Check if the provided activation is valid."""
    if isinstance(activation, str):
        return activation in activations.__dict__
    elif callable(activation):
        return True
    return False

def get_fit_and_compile_kwargs(kwargs) -> tuple[dict, dict]:
    """Parse kwargs and return separate dictionaries for fit and compile kwargs separately."""
    kwargs = copy.deepcopy(kwargs)
    compile_args = list(inspect.signature(Model.compile).parameters)
    compile_kwargs = {k: kwargs.pop(k) for k in dict(kwargs) if k in compile_args}
    fit_args = list(inspect.signature(Model.fit).parameters)
    fit_kwargs = {k: kwargs.pop(k) for k in dict(kwargs) if k in fit_args}
    return fit_kwargs, compile_kwargs
