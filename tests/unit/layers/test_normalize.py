import keras.src.backend as k
import numpy as np
from absl.testing import parameterized
from keras.src import testing

from sofenn.layers import NormalizeLayer

SHAPE_1D = (10,)
SAMPLES = (100,)
PARAM_COMBOS = [
            {"testcase_name": "1D", "shape": SHAPE_1D},
            {"testcase_name": "2D", "shape": SAMPLES + SHAPE_1D},
        ]


class NormalizeLayerTest(testing.TestCase):

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_build_across_shape_dimensions(self, shape):
        init_kwargs = {
            "shape": shape,
        }
        values = NormalizeLayer(**init_kwargs)(k.KerasTensor(shape))

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, shape)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_normalize_basics(self, shape):
        self.run_layer_test(
            NormalizeLayer,
            init_kwargs={
                "shape": shape,
            },
            input_shape=shape,
            expected_output_shape=shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            supports_masking=False,
            assert_built_after_instantiation=True,
        )

    @parameterized.named_parameters(PARAM_COMBOS)
    def testing_input_tensor(self, shape):
        input_tensor = k.KerasTensor(shape=shape)
        values = NormalizeLayer(shape=shape)(input_tensor)

        self.assertIsInstance(values, k.KerasTensor)
        self.assertEqual(values.shape, input_tensor.shape)
        self.assertEqual(values.ndim, input_tensor.ndim)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_call_method(self, shape):
        input_tensor = k.convert_to_tensor(np.random.random(shape))
        layer = NormalizeLayer(shape=shape)
        output = layer.call(inputs=input_tensor)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, shape)

    def test_numpy_shape(self):
        # non-python int type shapes should be ok
        NormalizeLayer(shape=(np.int64(5),))

    def test_get_config(self):
        config = NormalizeLayer(shape=SHAPE_1D).get_config()

        self.assertTrue('name' in config)
        self.assertTrue(config['shape'] == SHAPE_1D)
        self.assertTrue(config['trainable'] == True)
