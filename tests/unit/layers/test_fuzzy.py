import keras
import keras.ops as K
import numpy as np
from absl.testing import parameterized
from keras.src import testing

from sofenn.layers import FuzzyLayer
from sofenn.utils.layers import replace_last_dim, remove_nones

NEURONS = 5
DEFAULT_DIM = 3
SHAPES = [
            {"testcase_name": "1D", "shape":        (5,)},
            {"testcase_name": "2D", "shape":        (4, 5)},
            {"testcase_name": "2D_w_None", "shape": (None, 5)},
            {"testcase_name": "3D", "shape":        (3, 4, 5)},
            {"testcase_name": "3D_w_None", "shape": (None, None, 5)},
            {"testcase_name": "4D", "shape":        (2, 3, 4, 5)},
]

class FuzzyLayerTest(testing.TestCase):

    @parameterized.named_parameters(SHAPES)
    def test_input_shapes(self, shape):
        input_tensor = K.convert_to_tensor(np.random.random(remove_nones(shape, DEFAULT_DIM)))
        layer = FuzzyLayer(neurons=NEURONS)
        output = layer.call(inputs=input_tensor)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, replace_last_dim(remove_nones(shape, DEFAULT_DIM), NEURONS))


    def test_invalid_neurons(self):
        for invalid_neuron in [0, -1]:
            with self.assertRaises(ValueError):
                FuzzyLayer(neurons=invalid_neuron)

    @parameterized.named_parameters(SHAPES)
    def test_build_across_shape_dimensions(self, shape):
        init_kwargs = {
            'neurons': NEURONS,
            'initializer_centers': 'uniform',
            'initializer_sigmas': 'ones'
        }
        values = FuzzyLayer(**init_kwargs)(keras.KerasTensor(shape))

        self.assertIsInstance(values, keras.KerasTensor)
        self.assertEqual(values.shape, replace_last_dim(shape, NEURONS))

    @parameterized.named_parameters(SHAPES)
    def test_fuzzy_basics(self, shape):
        self.run_layer_test(
            FuzzyLayer,
            init_kwargs={
                'neurons': NEURONS,
                'initializer_centers': 'uniform',
                'initializer_sigmas': 'ones'
            },
            call_kwargs={
                'inputs': keras.KerasTensor(shape=shape)
            },
            expected_output_shape=replace_last_dim(shape, NEURONS),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            supports_masking=False,
            assert_built_after_instantiation=False,
        )

    @parameterized.named_parameters(SHAPES)
    def testing_input_tensor(self, shape):
        input_tensor = keras.KerasTensor(shape=shape)
        values = FuzzyLayer(neurons=NEURONS)(input_tensor)

        self.assertIsInstance(values, keras.KerasTensor)
        self.assertEqual(values.shape, replace_last_dim(input_tensor.shape, NEURONS))
        self.assertEqual(values.ndim, input_tensor.ndim)

    @parameterized.named_parameters(SHAPES)
    def test_call_method(self, shape):
        input_tensor = K.convert_to_tensor(np.random.random(remove_nones(shape, DEFAULT_DIM)))
        layer = FuzzyLayer(neurons=NEURONS)
        output = layer.call(inputs=input_tensor)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, replace_last_dim(remove_nones(shape, DEFAULT_DIM), NEURONS))

    def test_get_config(self):
        config = FuzzyLayer(neurons=NEURONS).get_config()

        self.assertTrue('name' in config)
        self.assertTrue(config['neurons'] == NEURONS)
        self.assertTrue('features' not in config)
        self.assertTrue('initializer_centers' in config)
        self.assertTrue('initializer_sigmas' in config)
        self.assertTrue(config['trainable'] == True)
