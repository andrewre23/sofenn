import keras.src.backend as k
import numpy as np
from absl.testing import parameterized
from keras.src import testing

from sofenn.layers import OutputLayer

SHAPE_1D = (10,)
SAMPLES = (100,)
OUTPUT_DIM = (1,)
PARAM_COMBOS = [
            {"testcase_name": "1D", "shape": SHAPE_1D},
            {"testcase_name": "2D", "shape": SAMPLES + SHAPE_1D},
            {"testcase_name": "with_None", "shape": (None,) + SHAPE_1D},
]


class OutputLayerTest(testing.TestCase):

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_build_across_shape_dimensions(self, shape):
        values = OutputLayer()(k.KerasTensor(shape))

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape[-1:], OUTPUT_DIM)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_output_basics(self, shape):
        # if None in shape:
        #     fixed_shape = shape[1:]
        self.run_layer_test(
            OutputLayer,
            init_kwargs={},
            input_shape=shape[1:] if None in shape else shape,
            expected_output_shape=OUTPUT_DIM if None in shape else shape[:-1] + OUTPUT_DIM,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            supports_masking=False,
            assert_built_after_instantiation=True,
        )

    @parameterized.named_parameters(PARAM_COMBOS)
    def testing_input_tensor(self, shape):
        input_tensor = k.KerasTensor(shape=shape)
        values = OutputLayer()(input_tensor)

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, input_tensor.shape[:-1] + OUTPUT_DIM)
        self.assertEqual(values.ndim, input_tensor.ndim)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_call_method(self, shape):
        if None in shape:
            fixed_shape = shape[1:]
        else:
            fixed_shape = shape
        input_tensor = k.convert_to_tensor(np.random.random(fixed_shape))
        layer = OutputLayer()
        output = layer.call(inputs=input_tensor)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, fixed_shape[:-1] + OUTPUT_DIM)

    def test_get_config(self):
        config = OutputLayer().get_config()

        self.assertTrue('name' in config)
        self.assertTrue(config['trainable'] == True)
