import copy
import pickle
import tempfile

import keras
import numpy
import pytest
from absl.testing import parameterized
from keras.callbacks import ProgbarLogger
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.optimizers import RMSprop, Adam
from keras.src import testing

from sofenn import FuzzyNetwork
from sofenn.callbacks import FuzzyWeightsInitializer
from sofenn.layers import FuzzyLayer, NormalizeLayer, WeightedLayer, OutputLayer
from sofenn.losses import CustomLoss
from tests.testing_utils import PROBLEM_DEFAULTS, PROBLEM_TYPES, DATA_DIR, \
    _init_params, _compile_params, _get_training_data, _load_saved_model


@pytest.mark.requires_trainable_backend
class FuzzyNetworkTest(testing.TestCase):

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            FuzzyNetwork(name='Neurons < 1', **_init_params(neurons=0))

        with self.assertRaises(ValueError):
            FuzzyNetwork(name='Number of classes < 1', **_init_params(num_classes=0))

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_basic_flow(self, problem_type):
        defaults = copy.deepcopy(PROBLEM_DEFAULTS[problem_type])
        input_shape = (defaults['samples'], defaults['features'])
        output_shape = (defaults['samples'], defaults['num_classes'])
        defaults.pop('features', None)

        model = FuzzyNetwork(**_init_params(problem_type))
        self.assertEqual(len(model.layers), 4)
        # until the model sees example data, then the model will not be built and weights will not be added
        self.assertFalse(model.built)
        self.assertEqual(len(model.weights), 0)

        # Test eager call
        x = numpy.random.random(input_shape)
        y = model(x)
        self.assertEqual(type(model), FuzzyNetwork)
        self.assertEqual(y.shape, output_shape)

        # Test symbolic call
        x = keras.KerasTensor(input_shape)
        y = model(x)
        self.assertEqual(y.shape, output_shape)

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_serialization(self, problem_type):
        model = FuzzyNetwork(name='Serialization test', **_init_params(problem_type))
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

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_saving_model(self, problem_type):
        epochs = 1
        X_train, X_test, y_train, y_test = _get_training_data(problem_type)

        trained_model = FuzzyNetwork(name='ModelFitTest',**_init_params(problem_type))
        trained_model.compile(**_compile_params(problem_type))
        trained_model.fit(X_train, y_train, epochs=epochs)

        with tempfile.TemporaryDirectory() as temp_directory:
            trained_model.save(temp_directory + 'model.keras')

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_functional_properties(self, problem_type):
        defaults = copy.deepcopy(PROBLEM_DEFAULTS[problem_type])
        model = FuzzyNetwork(name='Functional properties test', **_init_params(problem_type))
        self.assertEqual(model.features, defaults['features'])

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_pickleable(self, problem_type):
        model = FuzzyNetwork(**_init_params(problem_type))
        result = pickle.loads(pickle.dumps(model))
        self.assertEqual(len(result.layers), 4)

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_hasattr(self, problem_type):
        model = FuzzyNetwork(name='Attribute test', **_init_params(problem_type))
        self.assertTrue(hasattr(model, 'input_shape'))
        self.assertTrue(hasattr(model, 'neurons'))
        self.assertTrue(hasattr(model, 'num_classes'))
        self.assertTrue(hasattr(model, 'inputs'))

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_compile(self, problem_type):
        model = FuzzyNetwork(name=f'Compile as {problem_type}', **_init_params(problem_type))
        model.compile(**_compile_params(problem_type))
        self.assertFalse(model.built)

    def test_fit_classification(self):
        X_train, X_test, y_train, y_test = _get_training_data('classification')

        trained_model = FuzzyNetwork(name='ClassificationModelFitTest', **_init_params('classification'))
        trained_model.compile(**_compile_params('classification'))
        trained_model.fit(X_train, y_train, epochs=10)
        #trained_model.save(DATA_DIR / 'models/iris_classification.keras')
        loaded_model = _load_saved_model('classification', deep=False)

        # deep trained model
        #trained_model.fit(X_train, y_train, epochs=250)
        #trained_model.save(DATA_DIR / 'models/iris_classification-deep.keras')
        #loaded_model = _load_saved_model('classification', deep=True)

        self.assertTrue(numpy.allclose(trained_model.predict(X_test), loaded_model.predict(X_test)))

    # def test_fit_logistic_regression(self):
    #     X_train, X_test, y_train, y_test = _get_training_data('logistic_regression')
    #
    #     trained_model = FuzzyNetwork(
    #         name='LogisticRegressionWithDefaults',
    #         **_init_params('logistic_regression')
    #     )
    #     trained_model.compile(**_compile_params('logistic_regression'))
    #     trained_model.fit(X_train, y_train, epochs=1)
    #
    #     trained_model = FuzzyNetwork(
    #         name='LogisticRegressionModelFitTest',
    #         **_init_params('logistic_regression')
    #     )
    #     trained_model.compile(
    #         **_compile_params(
    #             'logistic_regression',
    #             loss=BinaryCrossentropy(),
    #             optimizer=Adam(learning_rate=0.1)
    #         )
    #     )
    #     trained_model.fit(X_train, y_train, epochs=25)
    #     #trained_model.save(DATA_DIR / f'models/logistic_regression.keras')
    #     loaded_model = _load_saved_model('logistic_regression', deep=False)
    #
    #     # deep trained model
    #     #trained_model.fit(X_train, y_train, epochs=250)
    #     #trained_model.save(DATA_DIR / f'models/logistic_regression-deep.keras')
    #     #loaded_model = _load_saved_model('logistic_regression', deep=True)
    #
    #     self.assertTrue(numpy.allclose(trained_model.predict(X_test), loaded_model.predict(X_test)))

    def test_fit_regression(self):
        X_train, X_test, y_train, y_test = _get_training_data('regression')

        trained_model = FuzzyNetwork(name='RegressionWithDefaults',**_init_params('regression'))
        trained_model.compile(**_compile_params('regression'))
        trained_model.fit(X_train, y_train, epochs=1)

        trained_model = FuzzyNetwork(name='RegressionModelFitTest',**_init_params('regression'))
        trained_model.compile(
            **_compile_params(
                'regression',
                loss=MeanSquaredError(),
                optimizer=RMSprop(learning_rate=0.1)
            )
        )
        trained_model.fit(X_train, y_train, epochs=25)
        #trained_model.save(DATA_DIR / f'models/regression.keras')
        loaded_model = _load_saved_model('regression', deep=False)

        # deep trained model
        #trained_model.fit(X_train, y_train, epochs=250)
        #trained_model.save(DATA_DIR / f'models/regression-deep.keras')
        #loaded_model = _load_saved_model('regression', deep=True)

        self.assertTrue(numpy.allclose(trained_model.predict(X_test), loaded_model.predict(X_test)))

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_fit_callbacks(self, problem_type):
        epochs = 1
        X_train, X_test, y_train, y_test = _get_training_data(problem_type)

        model = FuzzyNetwork(name='AppendToOtherCallbacks', **_init_params(problem_type))
        model.compile(**_compile_params(problem_type))
        model.fit(X_train, y_train, epochs=epochs, callbacks=[ProgbarLogger()])

        model = FuzzyNetwork(name='InitializerCallbackAlreadyProvided', **_init_params(problem_type))
        model.compile(**_compile_params(problem_type))
        model.fit(X_train, y_train, epochs=epochs, callbacks=[
            FuzzyWeightsInitializer(
                sample_data=X_train,
                random_sample=False
            )
        ])

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_summary(self, problem_type):
        model = FuzzyNetwork(name=f'Summary test: {problem_type.capitalize()}', **_init_params(problem_type))
        self.assertFalse(model.built)
        model.summary()
        self.assertFalse(model.built)

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_from_config(self, problem_type):
        config = _init_params(problem_type, name='From config')
        model = FuzzyNetwork.from_config(config)
        for key, value in config.items():
            self.assertEqual(model.__getattribute__(key), value)
