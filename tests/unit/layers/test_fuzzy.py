import keras.src.backend as k
import numpy as np
from absl.testing import parameterized
from keras.src import testing

from sofenn.layers import FuzzyLayer
from sofenn.utils.layers import get_fuzzy_output_shape

SHAPE_1D = (10,)
SAMPLES = (100,)
NEURONS = 5
PARAM_COMBOS = [
            {"testcase_name": "1D", "shape": SHAPE_1D},
            {"testcase_name": "2D", "shape": SAMPLES + SHAPE_1D},
        ]


class FuzzyLayerTest(testing.TestCase):

    def test_invalid_neurons(self):
        for invalid_neuron in [0, -1]:
            with self.assertRaises(ValueError):
                FuzzyLayer(shape=SHAPE_1D, neurons=invalid_neuron)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_build_across_shape_dimensions(self, shape):
        init_kwargs = {
            'shape': shape,
            'neurons': NEURONS,
            'initializer_centers': 'uniform',
            'initializer_sigmas': 'ones'
        }
        values = FuzzyLayer(**init_kwargs)(k.KerasTensor(shape))

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, get_fuzzy_output_shape(input_shape=shape, neurons=NEURONS))

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_fuzzy_basics(self, shape):
        self.run_layer_test(
            FuzzyLayer,
            init_kwargs={
                'shape': shape,
                'neurons': NEURONS,
                'initializer_centers': 'uniform',
                'initializer_sigmas': 'ones'
            },
            input_shape=shape,
            expected_output_shape=get_fuzzy_output_shape(input_shape=shape, neurons=NEURONS),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            supports_masking=False,
            assert_built_after_instantiation=False,
        )

    @parameterized.named_parameters(PARAM_COMBOS)
    def testing_input_tensor(self, shape):
        input_tensor = k.KerasTensor(shape=shape)
        values = FuzzyLayer(shape=shape, neurons=NEURONS)(input_tensor)

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, get_fuzzy_output_shape(input_shape=input_tensor.shape, neurons=NEURONS))
        self.assertEqual(values.ndim, input_tensor.ndim)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_call_method(self, shape):
        input_tensor = k.convert_to_tensor(np.random.random(shape))
        layer = FuzzyLayer(shape=shape, neurons=NEURONS)
        output = layer.call(inputs=input_tensor)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, get_fuzzy_output_shape(input_shape=shape, neurons=NEURONS))

    def test_numpy_shape(self):
        # non-python int type shapes should be ok
        FuzzyLayer(shape=(np.int64(5),), neurons=NEURONS)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_get_config(self, shape):
        config = FuzzyLayer(shape=shape, neurons=NEURONS).get_config()

        self.assertTrue('name' in config)
        self.assertTrue(config['neurons'] == NEURONS)
        self.assertTrue('features' not in config)
        self.assertTrue(config['shape'] == get_fuzzy_output_shape(input_shape=shape, neurons=NEURONS))
        self.assertTrue(config['trainable'] == True)
