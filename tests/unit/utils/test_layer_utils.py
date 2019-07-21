from keras.models import Model

from sofenn.utils.layers import remove_nones, replace_last_dim, make_2d, is_valid_activation, parse_function_kwargs


def test_remove_nones():
    default_dim = 2
    test_cases = {
        (3,)           : (3,),
        (3, 7)         : (3, 7),
        (3, 7, 2)      : (3, 7, 2),
        (None, 3)      : (default_dim, 3),
        (None, None, 3): (default_dim, default_dim, 3),
    }
    for shape, output_shape in test_cases.items():
        assert remove_nones(shape=shape, value=default_dim) == output_shape

def test_replace_last_dim():
    test_cases = {
        ((3,),            5): (5,),
        ((3, 7),          5): (3, 5),
        ((3, 7, 2),       5): (3, 7, 5),
        ((None, 3),       5): (None, 5),
        ((None, None, 3), 5): (None, None, 5),
    }
    for (input_shape, new_dim), output_shape in test_cases.items():
        assert replace_last_dim(input_shape, new_dim) == output_shape

def test_make_2d():
    assert make_2d((3,)) == (1, 3)
    assert make_2d((3, 1)) == (3, 1)

def test_is_valid_activation():
    from keras.activations import linear, softmax, sigmoid, relu

    for activation in [linear, softmax, sigmoid, relu]:
        assert is_valid_activation(activation)
        assert is_valid_activation(activation.__name__)

    assert ~is_valid_activation('invalid activation function')
    assert ~is_valid_activation(lambda x: x)

def test_separating_fit_and_compile_kwargs():
    starting_kwargs = {
        'name': 'kwargs extraction test',
        'neurons': 4,
        'input_shape': (None, None),
        'epochs': 3,
        'verbose': 2,
        'batch_size': 10,
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'metrics': 'categorical_accuracy',

    }
    target_fit_kwargs = {k: v for k,v in starting_kwargs.items() if k in ['epochs', 'verbose', 'batch_size']}
    target_compile_kwargs = {k: v for k,v in starting_kwargs.items() if k in ['optimizer', 'loss', 'metrics']}

    fit_kwargs = parse_function_kwargs(starting_kwargs, Model.fit)
    compile_kwargs = parse_function_kwargs(starting_kwargs, Model.compile)

    assert fit_kwargs == target_fit_kwargs
    assert compile_kwargs == target_compile_kwargs
