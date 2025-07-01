import copy
import inspect
from typing import Union, Callable

from keras import activations


def remove_nones(shape: tuple, value):
    """Remove none values from input shape."""
    return tuple([s if s is not None else value for s in shape]) if None in shape else shape

def replace_last_dim(shape: tuple, value: int):
    """Replace the last dimension of input shape."""
    new_shape = list(shape)
    new_shape[-1] = value
    return tuple(new_shape)

def make_2d(shape: tuple):
    """Prepend 1 dimension for batch size 1 to input shape if input is 1D."""
    return (1,) + shape if len(shape) == 1 else shape

def is_valid_activation(activation: Union[str, Callable]):
    """Check if the provided activation is valid."""
    if isinstance(activation, str):
        return activation in activations.__dict__
    elif callable(activation) and activation.__name__ in activations.__dict__:
        return True
    return False

def parse_function_kwargs(kwargs: dict, f: Callable) -> dict:
    """Parse kwargs and return separate dictionaries for function kwargs separately."""
    kwargs = copy.deepcopy(kwargs)
    args = list(inspect.signature(f).parameters)
    return {k: kwargs.pop(k) for k in dict(kwargs) if k in args}
