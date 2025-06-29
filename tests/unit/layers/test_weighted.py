import keras.src.backend as k
import numpy as np
from absl.testing import parameterized
from keras.src import testing

from sofenn.layers import WeightedLayer
from sofenn.utils.layers import remove_nones

DEFAULT_DIM = 3
SHAPES = [
            {"testcase_name": "1D", "shape":        [(5,),      (3,)]},
            {"testcase_name": "1D_w_None", "shape": [(None,),   (None,)]},
            {"testcase_name": "2D", "shape":        [(4, 5),    (4, 2)]},
            {"testcase_name": "2D_w_None", "shape": [(None, 5), (None, 2)]},
]


class WeightedLayerTest(testing.TestCase):

    @parameterized.named_parameters(SHAPES)
    def test_build_across_shape_dimensions(self, shape):
        init_kwargs = {
            "input_shape": shape,
        }
        features_shape, neuron_shape = shape
        values = WeightedLayer(**init_kwargs)([k.KerasTensor(features_shape), k.KerasTensor(neuron_shape)])

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, neuron_shape)

    @parameterized.named_parameters(SHAPES)
    def test_weighted_basics(self, shape):
        features_shape, neuron_shape = shape
        features_tensor = k.KerasTensor(shape=features_shape)
        neurons_tensor = k.KerasTensor(shape=neuron_shape)

        self.run_layer_test(
            WeightedLayer,
            init_kwargs={
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

    @parameterized.named_parameters(SHAPES)
    def testing_input_tensor(self, shape):
        print(f'shape: {shape}')
        features_shape, neuron_shape = shape
        features_tensor = k.KerasTensor(shape=features_shape)
        neurons_tensor = k.KerasTensor(shape=neuron_shape)
        values = WeightedLayer()([features_tensor, neurons_tensor])

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, neurons_tensor.shape)
        self.assertEqual(values.ndim, neurons_tensor.ndim)

    @parameterized.named_parameters(SHAPES)
    def test_call_method(self, shape):
        features_shape, neuron_shape = shape
        features_tensor = k.convert_to_tensor(np.random.random(remove_nones(features_shape, DEFAULT_DIM)))
        neurons_tensor = k.convert_to_tensor(np.random.random(remove_nones(neuron_shape, DEFAULT_DIM)))
        layer = WeightedLayer()
        output = layer.call(inputs=[features_tensor, neurons_tensor])

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, remove_nones(neuron_shape, DEFAULT_DIM))

    @parameterized.named_parameters(SHAPES)
    def test_get_config(self, shape):
        config = WeightedLayer().get_config()

        self.assertTrue('name' in config)
        self.assertTrue('initializer_a' in config)
        self.assertTrue(config['trainable'] == True)
