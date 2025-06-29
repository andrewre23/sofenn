import keras
import keras.ops as k
import numpy
from absl.testing import parameterized
from keras.src import testing

from sofenn.layers import OutputLayer
from sofenn.utils.layers import remove_nones, replace_last_dim, make_2d
from tests.testing_utils import PROBLEM_TYPES, PROBLEM_DEFAULTS

NEURONS = 10
DEFAULT_DIM = 2
TARGET_CLASSES = 3
SHAPES = [
            {"testcase_name": "1D", "shape":        (5,)},
            {"testcase_name": "2D", "shape":        (4, 5)},
            {"testcase_name": "2D_w_None", "shape": (None, 5)},
]


class OutputLayerTest(testing.TestCase):

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            OutputLayer(shape=(1,), target_classes=-1)

        with self.assertRaises(ValueError):
            OutputLayer(shape=(3, 2, 1), target_classes=1, activation='invalid')


    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_problem_types(self, problem_type):
        self.assertTrue(OutputLayer(
            num_classes=PROBLEM_DEFAULTS[problem_type]['num_classes']
        ))

    @parameterized.named_parameters(SHAPES)
    def test_build_across_shape_dimensions(self, shape):
        for problem_type in PROBLEM_DEFAULTS.keys():
            num_classes = PROBLEM_DEFAULTS[problem_type]['num_classes']
            init_kwargs = {
                "num_classes": num_classes,
            }
            values = OutputLayer(**init_kwargs)(keras.KerasTensor(shape))

            self.assertIsInstance(values, keras.KerasTensor)
            self.assertEqual(values.shape[-1], num_classes)

    @parameterized.named_parameters(SHAPES)
    def test_output_basics(self, shape):
        for problem_type in PROBLEM_DEFAULTS.keys():
            num_classes = PROBLEM_DEFAULTS[problem_type]['num_classes']
            self.run_layer_test(
                OutputLayer,
                init_kwargs={
                    'num_classes': num_classes,
                },
                call_kwargs={'inputs': keras.KerasTensor(shape=shape)},
                expected_output_shape=make_2d(replace_last_dim(shape, num_classes)),
                expected_num_trainable_weights=0,
                expected_num_non_trainable_weights=0,
                supports_masking=False,
                run_mixed_precision_check=False,
                assert_built_after_instantiation=False,
            )

    @parameterized.named_parameters(SHAPES)
    def testing_input_tensor(self, shape):
        for problem_type in PROBLEM_DEFAULTS.keys():
            num_classes = PROBLEM_DEFAULTS[problem_type]['num_classes']
            input_tensor = keras.KerasTensor(shape=shape)
            values = OutputLayer(
                num_classes=num_classes,
            )(input_tensor)

            self.assertIsInstance(values, keras.KerasTensor)
            self.assertEqual(values.shape,replace_last_dim(shape, num_classes))
            self.assertEqual(values.ndim, input_tensor.ndim)

    @parameterized.named_parameters(SHAPES)
    def test_call_method(self, shape):
        for problem_type in PROBLEM_DEFAULTS.keys():
            num_classes = PROBLEM_DEFAULTS[problem_type]['num_classes']
            input_shape = remove_nones(shape, DEFAULT_DIM)
            input_tensor = k.convert_to_tensor(numpy.random.random(input_shape))
            layer = OutputLayer(
                num_classes=num_classes,
                activation=PROBLEM_DEFAULTS[problem_type]['activation'],
            )
            output = layer.call(inputs=input_tensor)

            self.assertIsNotNone(output)
            self.assertEqual(
                output.shape,
                make_2d(replace_last_dim(input_shape, num_classes))
            )

    def test_get_config(self):
        for problem_type in PROBLEM_DEFAULTS.keys():
            activation = 'sigmoid'
            num_classes = PROBLEM_DEFAULTS[problem_type]['num_classes']
            config = OutputLayer(
                num_classes=num_classes,
                activation=activation
            ).get_config()

            self.assertTrue('name' in config)
            self.assertTrue(config['num_classes'] == num_classes)
            self.assertTrue(config['activation'] == 'sigmoid')
            self.assertTrue(config['trainable'] == True)
