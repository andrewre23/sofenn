import copy
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path

import keras
import keras.src.backend as K
import numpy
import pandas
import pytest
from keras.api.callbacks import ProgbarLogger
from keras.src import testing
from keras.src.losses import MeanSquaredError
from keras.src.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from sofenn import FuzzyNetwork
from sofenn.callbacks import FuzzyWeightsInitializer
from sofenn.layers import FuzzyLayer, NormalizeLayer, WeightedLayer, OutputLayer
from sofenn.losses import CustomLoss

DATA_DIR = Path(__file__).parent / 'data'
DEFAULTS = {
    'features': 4,
    'neurons': 3,
    'problem_type': 'classification',
    'target_classes': 3,
    'samples': 10
}


@lru_cache(maxsize=None)
def _params(**kwargs):
    params = copy.deepcopy(DEFAULTS)
    params.pop('samples')
    for key, value in kwargs.items():
        params[key] = value
    return params

@lru_cache(maxsize=None)
def _get_training_data():
    features = pandas.read_csv(DATA_DIR / 'iris/features.csv')
    target = pandas.read_csv(DATA_DIR / 'iris/target.csv')
    return train_test_split(features.values, target.values, test_size=0.1, random_state=23)


@pytest.mark.requires_trainable_backend
class FuzzyNetworkTest(testing.TestCase):

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='Neurons < 1',
                neurons=0
            ))

        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='Invalid problem type',
                problem_type='guessing_game'
            ))

    def test_inputs_for_classification(self):
        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='No target classes provided',
                target_classes=None
            ))

        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='Target classes < 2',
                target_classes=1
            ))

    def test_inputs_for_logistic_regression(self):
        with self.assertLogs(level='WARNING'):
            FuzzyNetwork(**_params(
                name='No target classes provided',
                target_classes=1,
                problem_type='logistic_regression'
            ))

    # def test_inputs_for_regression(self):
    #     with self.assertRaises(ValueError):
    #         FuzzyNetwork(**_params(
    #             name='None Yet',
    #             problem_type='regression'
    #         ))

    def test_init_with_features_and_input_shape(self):
        FuzzyNetwork(**_params(
            name='Only features provided',
            features=DEFAULTS['features']
        ))

        params = _params(
            name='Only input shape provided',
            input_shape=(DEFAULTS['features'],)
        )
        params.pop('features')
        FuzzyNetwork(**params)

        FuzzyNetwork(**_params(
            name='Input shape and feature agree',
            features=DEFAULTS['features'],
            input_shape=(DEFAULTS['features'],)
        ))

        FuzzyNetwork(**_params(
            name='Input shape and features parameters agree, and samples placeholder added to input shape',
            features=DEFAULTS['features'],
            input_shape=(None, DEFAULTS['features'])
        ))

        with self.assertRaises(ValueError):
            params = _params(name='Neither input shape or features are provided')
            params.pop('features')
            FuzzyNetwork(**params)

        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name="Input shape and features don't agree",
                features=DEFAULTS['features'],
                input_shape=(DEFAULTS['features'] + 1,)
            ))

        with self.assertRaises(ValueError):
            FuzzyNetwork(**_params(
                name='Features < 1',
                features=0,
            ))

    def test_basic_flow(self):
        model = FuzzyNetwork(
            features=DEFAULTS['features'],
            neurons=DEFAULTS['neurons'],
            problem_type=DEFAULTS['problem_type'],
            target_classes=DEFAULTS['target_classes']
        )
        self.assertEqual(len(model.layers), 4)
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
        model = FuzzyNetwork(**_params(name='Serialization test'))
        revived = self.run_class_serialization_test(
            model,
            custom_objects={
                'FuzzyLayer': FuzzyLayer,
                'NormalizationLayer': NormalizeLayer,
                'WeightedLayer': WeightedLayer,
                'OutputLayer': OutputLayer,
                'CustomLoss': CustomLoss,
                'FuzzyWeightsInitializer': FuzzyWeightsInitializer
            }
        )
        self.assertLen(revived.layers, 4)

    def test_saving_model(self):
        epochs = 1
        X_train, X_test, y_train, y_test = _get_training_data()

        trained_model = FuzzyNetwork(**_params(name='ModelFitTest'))
        trained_model.compile()
        trained_model.fit(X_train, y_train, epochs=epochs)

        with tempfile.TemporaryDirectory() as temp_directory:
            trained_model.save(temp_directory + 'model.keras')

    def test_functional_properties(self):
        model = FuzzyNetwork(**_params(name='Functional properties test'))

        self.assertEqual(model.input_shape, (None, DEFAULTS['features']))
        self.assertEqual(model.output_shape, (None, DEFAULTS['target_classes']))

    def test_pickleable(self):
        model = FuzzyNetwork(**_params(name='Pickleable test'))
        result = pickle.loads(pickle.dumps(model))
        assert len(result.layers) == 4

    def test_hasattr(self):
        model = FuzzyNetwork(**_params(name='Attribute test'))
        self.assertTrue(hasattr(model, "features"))
        self.assertTrue(hasattr(model, "input_shape"))
        self.assertTrue(hasattr(model, "output_shape"))
        self.assertTrue(hasattr(model, "neurons"))
        self.assertTrue(hasattr(model, "problem_type"))
        self.assertTrue(hasattr(model, "target_classes"))
        self.assertTrue(hasattr(model, "inputs"))

    def test_compile(self):
        model = FuzzyNetwork(**_params(
            name='Compile as classification',
            problem_type='classification'
        ))
        model.compile()
        self.assertFalse(model.built)

        model = FuzzyNetwork(**_params(
            name='Compile as regression',
            problem_type='regression'
        ))
        model.compile()
        self.assertFalse(model.built)

    def test_fit_classification(self):
        epochs = 10
        X_train, X_test, y_train, y_test = _get_training_data()

        trained_model = FuzzyNetwork(**_params(name='ClassificationModelFitTest'))
        trained_model.compile()
        trained_model.fit(X_train, y_train, epochs=epochs)
        #trained_model.save(DATA_DIR / 'models/iris_classification.keras')
        loaded_model = keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')

        # deep trained model
        #trained_model.fit(X_train, y_train, epochs=250)
        #trained_model.save(DATA_DIR / 'models/iris_classification-deep.keras')
        #loaded_model = keras.saving.load_model(DATA_DIR / 'models/iris_classification-deep.keras')

        assert numpy.allclose(trained_model.predict(X_test), loaded_model.predict(X_test))

    # def test_fit_logistic_regression(self):
    #     epochs = 10
    #     samples = 25
    #     features = 4
    #     #X_train = numpy.linspace(0, 100, 25)
    #     X_train = numpy.random.random((samples, features))
    #     noise = numpy.random.normal(0,.5, len(X_train))
    #     y_train = numpy.dot(X_train, [3, 1, 2, 1]) + noise
    #
    #     trained_model = FuzzyNetwork(
    #         name='LogisticRegressionModelFitTest',
    #         input_shape=X_train.shape,
    #         problem_type='logistic_regression',
    #         neurons=3
    #     )
    #     trained_model.compile(
    #         loss=BinaryCrossentropy(from_logits=False),
    #     )
    #     trained_model.fit(X_train, y_train, epochs=epochs)
    #     #trained_model.save(DATA_DIR / 'models/regression.keras')
    #
    #     #loaded_model = keras.saving.load_model(DATA_DIR / 'models/regression.keras')
    #     #assert numpy.allclose(trained_model.predict(X_test), loaded_model.predict(X_test))


    def test_fit_regression(self):
        samples = 250
        train_samples = 200
        features = DEFAULTS['features']
        x = numpy.random.random((samples, features))
        noise = numpy.random.normal(0,.5, len(x))
        y = numpy.dot(x, [8, 1, 3, 2]) + noise
        X_train, X_test = x[:train_samples], x[train_samples:]
        y_train, y_test = y[:train_samples], y[train_samples:]


        trained_model = FuzzyNetwork(**_params(
            name='RegressionWithDefaults',
            problem_type='regression'
        ))
        trained_model.compile()
        trained_model.fit(X_train, y_train, epochs=1)

        trained_model = FuzzyNetwork(
            name='RegressionModelFitTest',
            features=features,
            problem_type='regression',
            neurons=5
        )
        trained_model.compile(
            loss=MeanSquaredError(),
            optimizer=RMSprop(learning_rate=0.1),
        )
        trained_model.fit(X_train, y_train, epochs=25)
        #trained_model.save(DATA_DIR / 'models/regression.keras')

        loaded_model = keras.saving.load_model(DATA_DIR / 'models/regression.keras')
        assert numpy.allclose(trained_model.predict(X_test), loaded_model.predict(X_test))

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
        model = FuzzyNetwork(**_params(name='Summary test'))
        self.assertFalse(model.built)
        model.summary()
        self.assertFalse(model.built)

    def test_from_config(self):
        config = _params(name='From config')
        model = FuzzyNetwork.from_config(config)
        for key, value in config.items():
            assert model.__getattribute__(key) == value
