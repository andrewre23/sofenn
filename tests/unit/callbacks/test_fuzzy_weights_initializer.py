import copy

import numpy
import pytest
from absl.testing import parameterized
from keras.src import testing

from sofenn import FuzzyNetwork
from sofenn.callbacks import FuzzyWeightsInitializer
from tests.testing_utils import PROBLEM_DEFAULTS

SHAPES = [
            {"testcase_name": "1D", 'x_shape': (10,),   'y_shape': (10,)},
            {"testcase_name": "2D", 'x_shape': (10, 4), 'y_shape': (10, 3)},
]

# TODO: clean and format file
@pytest.mark.requires_trainable_backend
class FuzzyNetworkTest(testing.TestCase):

    @parameterized.named_parameters(SHAPES)
    def test_dimensions(self, x_shape, y_shape):
        for problem_type, defaults in PROBLEM_DEFAULTS.items():
            compile_defaults = copy.deepcopy(defaults['compile'])

            x = numpy.random.random(x_shape)
            y = numpy.random.random(y_shape)

            model = FuzzyNetwork(
                input_shape=x.shape,
                neurons=5,
                num_classes=y.shape[-1]
            )
            model.compile(
                run_eagerly=True,
                **{
                    key: [v() for v in val] if isinstance(val, list) else val()
                    for key, val in copy.deepcopy(compile_defaults).items()
                }
            )
            model.fit(x, y, epochs=1, callbacks=[FuzzyWeightsInitializer(sample_data=x)])

    def test_fail_on_non_fuzzy_layer(self):
        x = numpy.random.random((10, 4))
        y = numpy.random.random((10, 3))

        with (self.assertRaises(ValueError)):
            model = FuzzyNetwork(
                input_shape=x.shape,
                num_classes=y.shape[-1]
            )
            model.compile(run_eagerly=True)
            model.fit(x, y, epochs=1, callbacks=[
                (FuzzyWeightsInitializer(sample_data=x, layer_name='Normalize'))
            ])

    def test_less_samples_than_neurons(self):
        for problem_type, defaults in PROBLEM_DEFAULTS.items():
            compile_defaults = copy.deepcopy(defaults['compile'])
            x = numpy.random.random(3)
            y = numpy.random.random(3)

            model = FuzzyNetwork(
                input_shape=x.shape,
                num_classes=y.shape[-1]
            )
            model.compile(
                run_eagerly=True,
                **{
                    key: [v() for v in val] if isinstance(val, list) else val()
                    for key, val in copy.deepcopy(compile_defaults).items()
                }
            )
            model.fit(x, y, epochs=1, callbacks=[
                (FuzzyWeightsInitializer(sample_data=x, layer_name='FuzzyRules'))
            ])

    def test_randomly_sampling(self):
        for problem_type, defaults in PROBLEM_DEFAULTS.items():
            compile_defaults = copy.deepcopy(defaults['compile'])

            x = numpy.random.random((10, 5))
            y = numpy.random.random((10, 3))

            for random_sample in [True, False]:
                model = FuzzyNetwork(
                    input_shape=x.shape,
                    num_classes=y.shape[-1]
                )
                model.compile(
                    run_eagerly=True,
                    **{
                        key: [v() for v in val] if isinstance(val, list) else val()
                        for key, val in copy.deepcopy(compile_defaults).items()
                    }
                )
                model.fit(x, y, epochs=1, callbacks=[
                    (FuzzyWeightsInitializer(sample_data=x, layer_name='FuzzyRules', random_sample=random_sample))
                ])
