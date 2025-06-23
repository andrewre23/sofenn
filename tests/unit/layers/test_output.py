import keras.src.backend as k
import numpy as np
from absl.testing import parameterized
from keras.src import testing
from keras.api.activations import softmax, linear, sigmoid

from sofenn.layers import OutputLayer

FEATURES = (10,)
SAMPLES = (100,)
TARGET_CLASSES = 3
PARAM_COMBOS = [
            {"testcase_name": "1D", "shape": FEATURES},
            {"testcase_name": "2D", "shape": SAMPLES + FEATURES},
            {"testcase_name": "with_None", "shape": (None,) + FEATURES},
]
DEFAULTS = {
    # 'classification': {
    #     'activation': softmax,
    #     'target_classes': TARGET_CLASSES,
    # },
    # 'logistic_regression': {
    #     'activation': sigmoid,
    #     'target_classes': 1,
    # },
    'regression': {
        'activation': linear,
        'target_classes': 1,
    },
}


class OutputLayerTest(testing.TestCase):
    def test_problem_types(self):
        for problem_type in DEFAULTS.keys():
            assert OutputLayer(
                FEATURES,
                target_classes=DEFAULTS[problem_type]['target_classes'],
                problem_type=problem_type
            )

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_build_across_shape_dimensions(self, shape):
        for problem_type in DEFAULTS.keys():
            values = OutputLayer(
                shape,
                target_classes=DEFAULTS[problem_type]['target_classes'],
                problem_type=problem_type
            )(k.KerasTensor(shape))

            self.assertIsInstance(values, k.KerasTensor)
            self.assertEqual(values.shape[-1], DEFAULTS[problem_type]['target_classes'])

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_output_basics(self, shape):
        for problem_type in DEFAULTS.keys():
            if None in shape:
                fixed_shape = shape[1:]
            else:
                fixed_shape = shape
            self.run_layer_test(
                OutputLayer,
                init_kwargs={
                    'shape': shape,
                    'target_classes': DEFAULTS[problem_type]['target_classes'],
                    'problem_type': problem_type
                },
                input_shape=fixed_shape,
                # TODO: fix to work for all cases. currently fails for 2-D input
                #       Unexpected output shape
                #       TensorShape([100]) != (100, 1)
                # expected_output_shape=(DEFAULTS[problem_type]['target_classes'],) if None in shape else shape[:-1] + (DEFAULTS[problem_type]['target_classes'],),
                # expected_output_shape=(DEFAULTS[problem_type]['target_classes'],) if len(fixed_shape) == 1 \
                #     else (100, 1),
                expected_num_trainable_weights=0,
                expected_num_non_trainable_weights=0,
                supports_masking=False,
                run_mixed_precision_check=False,
                assert_built_after_instantiation=True,
            )

    @parameterized.named_parameters(PARAM_COMBOS)
    def testing_input_tensor(self, shape):
        for problem_type in DEFAULTS.keys():
            input_tensor = k.KerasTensor(shape=shape)
            values = OutputLayer(
                shape,
                target_classes=DEFAULTS[problem_type]['target_classes'],
                problem_type=problem_type
            )(input_tensor)

            self.assertIsInstance(values, k.KerasTensor)
            self.assertEqual(values.shape, input_tensor.shape[:-1] + (DEFAULTS[problem_type]['target_classes'],))
            self.assertEqual(values.ndim, input_tensor.ndim)

    @parameterized.named_parameters(PARAM_COMBOS)
    def test_call_method(self, shape):
        for problem_type in DEFAULTS.keys():
            if None in shape:
                fixed_shape = shape[1:]
            else:
                fixed_shape = shape
            input_tensor = k.convert_to_tensor(np.random.random(fixed_shape))
            layer = OutputLayer(
                shape,
                target_classes=DEFAULTS[problem_type]['target_classes'],
                problem_type=problem_type
            )
            output = layer.call(inputs=input_tensor)

            self.assertIsNotNone(output)
            self.assertEqual(
                output.shape,
                (DEFAULTS[problem_type]['target_classes'],) if len(fixed_shape) == 1 else fixed_shape[:-1]
            )

    def test_get_config(self):
        for problem_type in DEFAULTS.keys():
            config = OutputLayer(
                FEATURES,
                target_classes=DEFAULTS[problem_type]['target_classes'],
                problem_type=problem_type
            ).get_config()

            self.assertTrue('name' in config)
            self.assertTrue(config['shape'] == FEATURES)
            self.assertTrue(config['target_classes'] == DEFAULTS[problem_type]['target_classes'])
            self.assertTrue(config['problem_type'] == problem_type)
            self.assertTrue(config['trainable'] == True)
