import numpy
import pytest
from keras.src import testing

from sofenn import FuzzyNetwork
from sofenn.callbacks import FuzzyWeightsInitializer


@pytest.mark.requires_trainable_backend
class FuzzyNetworkTest(testing.TestCase):

    def test_fail_on_non_fuzzy_layer(self):
        x_train = numpy.random.random((10, 4))
        y_train = numpy.random.random((10, 3))

        with (self.assertRaises(ValueError)):
            model = FuzzyNetwork(
                input_shape=x_train.shape,
                target_shape=y_train.shape,
                target_classes=y_train.shape[-1]
            )
            model.compile(run_eagerly=True)
            model.fit(
                x_train,
                y_train,
                epochs=1,
                callbacks=[(
                    FuzzyWeightsInitializer(
                        sample_data=x_train,
                        layer_name='NotFuzzyLayer'
                    )
                )],
            )
