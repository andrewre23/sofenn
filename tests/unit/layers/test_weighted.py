import keras.src.backend as k
import numpy as np
from absl.testing import parameterized
from keras.src import testing

from sofenn.layers import WeightedLayer

FEATURES_1D = (12,)
NEURONS_1D = (5,)
SAMPLES = (100,)
SHAPE_1D = [FEATURES_1D, NEURONS_1D]
PARAM_COMBOS = [
            {"testcase_name": "1D", "shape": SHAPE_1D},
            {"testcase_name": "2D", "shape": [SAMPLES + features_or_neurons for features_or_neurons in SHAPE_1D]},
        ]


class WeightedLayerTest(testing.TestCase):

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_build_across_shape_dimensions(self, shape):
        init_kwargs = {
            "shape": shape,
        }
        features_shape, neuron_shape = shape
        values = WeightedLayer(**init_kwargs)([k.KerasTensor(features_shape), k.KerasTensor(neuron_shape)])

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, neuron_shape)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_weighted_basics(self, shape):
        features_shape, neuron_shape = shape
        features_tensor = k.KerasTensor(shape=features_shape)
        neurons_tensor = k.KerasTensor(shape=neuron_shape)
        self.run_layer_test(
            WeightedLayer,
            init_kwargs={
                'shape': shape,
                'initializer_a': 'uniform',
            },
            call_kwargs={
                'inputs': [features_tensor, neurons_tensor]
            },
            expected_output_shape=neuron_shape,
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            supports_masking=False,
            assert_built_after_instantiation=False,
        )

    @parameterized.named_parameters(PARAM_COMBOS)
    def testing_input_tensor(self, shape):
        features_shape, neuron_shape = shape
        features_tensor = k.KerasTensor(shape=features_shape)
        neurons_tensor = k.KerasTensor(shape=neuron_shape)
        values = WeightedLayer(shape=shape)([features_tensor, neurons_tensor])

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, neurons_tensor.shape)
        self.assertEqual(values.ndim, neurons_tensor.ndim)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_call_method(self, shape):
        features_shape, neuron_shape = shape
        features_tensor = k.convert_to_tensor(np.random.random(features_shape))
        neurons_tensor = k.convert_to_tensor(np.random.random(neuron_shape))
        layer = WeightedLayer(shape=shape)
        output = layer.call(inputs=[features_tensor, neurons_tensor])

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, neuron_shape)

    def test_numpy_shape(self):
        # non-python int type shapes should be ok
        WeightedLayer(
            shape=[
                (np.int64(12),),
                (np.int64(5),)
            ]
        )

    def test_get_config(self):
        config = WeightedLayer(shape=SHAPE_1D).get_config()

        self.assertTrue('name' in config)
        self.assertTrue(config['shape'] == SHAPE_1D)
        self.assertTrue('initializer_a' in config)
        self.assertTrue(config['trainable'] == True)
