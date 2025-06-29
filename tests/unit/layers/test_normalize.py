import keras.src.backend as k
import numpy as np
from absl.testing import parameterized
from keras.src import testing

from sofenn.layers import NormalizeLayer
from sofenn.utils.layers import remove_nones

DEFAULT_DIM = 3
SHAPES = [
            {"testcase_name": "1D", "shape":        (5,)},
            {"testcase_name": "2D", "shape":        (4, 5)},
            {"testcase_name": "2D_w_None", "shape": (None, 5)},
            {"testcase_name": "3D", "shape":        (3, 4, 5)},
            {"testcase_name": "3D_w_None", "shape": (None, None, 5)},
            {"testcase_name": "4D", "shape":        (2, 3, 4, 5)},
]


class NormalizeLayerTest(testing.TestCase):

    @parameterized.named_parameters(SHAPES)
    def test_build_across_shape_dimensions(self, shape):
        init_kwargs = {}
        values = NormalizeLayer(**init_kwargs)(k.KerasTensor(shape))

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, shape)

    @parameterized.named_parameters(SHAPES)
    def test_normalize_basics(self, shape):
        self.run_layer_test(
            NormalizeLayer,
            init_kwargs={},
            call_kwargs={
                'inputs': k.KerasTensor(shape=shape)
            },
            expected_output_shape=shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=1,
            supports_masking=False,
            assert_built_after_instantiation=False,
        )

    @parameterized.named_parameters(SHAPES)
    def testing_input_tensor(self, shape):
        input_tensor = k.KerasTensor(shape=shape)
        values = NormalizeLayer()(input_tensor)

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, input_tensor.shape)
        self.assertEqual(values.ndim, input_tensor.ndim)

    @parameterized.named_parameters(SHAPES)
    def test_call_method(self, shape):
        input_tensor = k.convert_to_tensor(np.random.random(remove_nones(shape, DEFAULT_DIM)))
        layer = NormalizeLayer()
        output = layer.call(inputs=input_tensor)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, remove_nones(shape, DEFAULT_DIM))

    def test_get_config(self):
        config = NormalizeLayer().get_config()

        self.assertTrue('name' in config)
        self.assertTrue(config['trainable'] == True)
