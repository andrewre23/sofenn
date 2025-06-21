import keras.src.backend as k

def get_fuzzy_output_shape(input_shape: tuple, neurons: int):
    """Get shape of fuzzy output based on input shape and neurons."""
    new_shape = list(input_shape)
    new_shape[-1] = neurons
    return k.standardize_shape(tuple(new_shape))


def fixed_shape(input_shape: tuple):
    """Return fixed shape of input, even when None provided as first input dimension."""
    if None in input_shape:
        return input_shape[1:]
    else:
        return input_shape
