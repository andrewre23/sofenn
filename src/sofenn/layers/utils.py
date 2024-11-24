import keras.src.backend as k

# TODO: move to sofenn.utils.layers
def get_fuzzy_output_shape(input_shape: tuple, neurons: int):
    new_shape = list(input_shape)
    new_shape[-1] = neurons
    return k.standardize_shape(tuple(new_shape))


def fixed_shape(input_shape: tuple):
    if None in input_shape:
        return input_shape[1:]
    else:
        return input_shape
