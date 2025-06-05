import copy
import pickle
from pathlib import Path

import keras.src.backend as K
import numpy
import numpy as np
import pandas
import pytest
from keras.api.callbacks import ProgbarLogger
from keras.src import testing
from sklearn.model_selection import train_test_split

from sofenn import FuzzyNetwork
from sofenn.callbacks import FuzzyWeightsInitializer

DATA_DIR = Path(__file__).parent / 'data'
DEFAULTS = {
    'features': 4,
    'neurons': 3,
    'problem_type': 'classification',
    'target_classes': 3,
    'samples': 10
}


def _params(**kwargs):
    params = copy.deepcopy(DEFAULTS)
    params.pop('samples')
    for key, value in kwargs.items():
        params[key] = value
    return params


def _get_training_data():
    features = pandas.read_csv(DATA_DIR / 'iris/features.csv')
    target = pandas.read_csv(DATA_DIR / 'iris/target.csv')
    return train_test_split(features.values, target.values, test_size=0.1, random_state=23)


@pytest.mark.requires_trainable_backend
class FuzzyNetworkTest(testing.TestCase):

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='Neurons < 1.',
                neurons=0
            ))

        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='Invalid problem type.',
                problem_type='guessing_game'
            ))

    def test_inputs_for_classification(self):
        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='No target classes provided.',
                target_classes=None
            ))

        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='Target classes < 2.',
                target_classes=1
            ))

    def test_init_with_features_and_input_shape(self):
        FuzzyNetwork(**_params(
            name='Only features provided.',
            features=DEFAULTS['features']
        ))

        params = _params(
            name='Only input shape provided.',
            input_shape=(DEFAULTS['features'],)
        )
        params.pop('features')
        FuzzyNetwork(**params)

        FuzzyNetwork(**_params(
            name='Input shape and feature agree.',
            features=DEFAULTS['features'],
            input_shape=(DEFAULTS['features'],)
        ))

        FuzzyNetwork(**_params(
            name='Input shape and features parameters agree, and samples placeholder added to input shape.',
            features=DEFAULTS['features'],
            input_shape=(None, DEFAULTS['features'])
        ))

        with self.assertRaises(ValueError):
            params = _params(name='Neither input shape or features are provided.')
            params.pop('features')
            FuzzyNetwork(**params)

        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name="Input shape and features don't agree.",
                features=DEFAULTS['features'],
                input_shape=(DEFAULTS['features'] + 1,)
            ))

        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='Features < 1.',
                features=0,
            ))

    def test_basic_flow(self):
        model = FuzzyNetwork(
            features=DEFAULTS['features'],
            neurons=DEFAULTS['neurons'],
            problem_type=DEFAULTS['problem_type'],
            target_classes=DEFAULTS['target_classes']
        )
        self.assertEqual(len(model.layers), 5)
        # until the model sees example data, then the model will not be built and weights will not be added
        self.assertFalse(model.built)
        self.assertEqual(len(model.weights), 0)

        # Test eager call
        x = numpy.random.random((DEFAULTS['samples'], DEFAULTS['features']))
        y = model(x)
        self.assertEqual(type(model), FuzzyNetwork)
        self.assertEqual(y.shape, (DEFAULTS['samples'], DEFAULTS['target_classes']))

        # Test symbolic call
        x = K.KerasTensor((DEFAULTS['samples'], DEFAULTS['features']))
        y = model(x)
        self.assertEqual(y.shape, (DEFAULTS['samples'], DEFAULTS['target_classes']))

    def test_serialization(self):
        model = FuzzyNetwork(**_params(name='Serialization test.'))
        revived = self.run_class_serialization_test(model)
        self.assertLen(revived.layers, 5)

    def test_functional_properties(self):
        model = FuzzyNetwork(**_params(name='Functional properties test.'))

        # self.assertEqual(model.inputs, INPUT DATA)
        #self.assertEqual(model.inputs, [FEATURES, NEURONS])
        #self.assertEqual(model.outputs, [model.layers[-1].output])
        self.assertEqual(model.input_shape, (None, DEFAULTS['features']))
        self.assertEqual(model.output_shape, (None, DEFAULTS['target_classes']))

    def test_pickleable(self):
        model = FuzzyNetwork(**_params(name='Pickleable test.'))
        result = pickle.loads(pickle.dumps(model))
        assert len(result.layers) == 5

    def test_hasattr(self):
        model = FuzzyNetwork(**_params(name='Attribute test.'))
        # TODO: add falses to check before/after model is compiled/fitted
        self.assertTrue(hasattr(model, "features"))
        self.assertTrue(hasattr(model, "input_shape"))
        self.assertTrue(hasattr(model, "output_shape"))
        self.assertTrue(hasattr(model, "neurons"))
        self.assertTrue(hasattr(model, "problem_type"))
        self.assertTrue(hasattr(model, "target_classes"))
        self.assertTrue(hasattr(model, "inputs"))

    def test_compile(self):
        model = FuzzyNetwork(**_params(
            name='Compile as classification.',
            problem_type='classification'
        ))
        model.compile()
        self.assertFalse(model.built)

        model = FuzzyNetwork(**_params(
            name='Compile as regression.',
            problem_type='regression'
        ))
        model.compile()
        self.assertFalse(model.built)

    def test_fit_classification(self):
        epochs = 10
        X_train, X_test, y_train, y_test = _get_training_data()

        trained_model = FuzzyNetwork(**_params(name='ModelFitTest'))
        trained_model.compile()
        trained_model.fit(X_train, y_train, epochs=epochs)
        #trained_model.save_weights(DATA_DIR / 'weights/classification.weights.h5')

        loaded_model = FuzzyNetwork(**_params(name='LoadedModel'))
        loaded_model.compile()
        loaded_model.fit(X_train, y_train, epochs=1)
        self.assertTrue(loaded_model.built)
        loaded_model.load_weights(DATA_DIR / 'weights/classification.weights.h5')

        assert np.allclose(trained_model.predict(X_test), loaded_model.predict(X_test))

    def test_fit_callbacks(self):
        epochs = 1
        X_train, X_test, y_train, y_test = _get_training_data()

        model = FuzzyNetwork(**_params(name='AppendToOtherCallbacks'))
        model.compile()
        model.fit(X_train, y_train, epochs=epochs, callbacks=[
            ProgbarLogger()
        ])

        model = FuzzyNetwork(**_params(name='InitializerCallbackAlreadyProvided'))
        model.compile()
        model.fit(X_train, y_train, epochs=epochs, callbacks=[
            FuzzyWeightsInitializer(
                sample_data=X_train,
                random_sample=False
            )
        ])

    def test_summary(self):
        model = FuzzyNetwork(**_params(name='Summary test.'))
        self.assertFalse(model.built)
        model.summary()
        self.assertFalse(model.built)
