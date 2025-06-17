import copy
import pickle
from pathlib import Path

import keras
import keras.src.backend as K
#import keras.src.saving
# TODO: remove numpy from dependency and replace with Keras functions
import numpy
import numpy as np
import pandas
import pytest
from keras.api.callbacks import ProgbarLogger
from keras.src import testing
from sklearn.model_selection import train_test_split

from sofenn import FuzzyNetwork, FuzzySelfOrganizer
from sofenn.callbacks import FuzzyWeightsInitializer

DATA_DIR = Path(__file__).parent / 'data'
DEFAULTS = {
    'features': 4,
    'neurons': 3,
    'problem_type': 'classification',
    'target_classes': 3,
    'samples': 10
}
# TODO: add dict of defaults for self organizer class and add to _params function option to add


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
class FuzzySelfOrganizerTest(testing.TestCase):

    def test_init_with_model(self):
        model = FuzzyNetwork(**_params(
            name='PreInitializedModel'
        ))
        assert model.get_config() == FuzzySelfOrganizer(model).model.get_config()

        FuzzySelfOrganizer(
            name='InputShapeAndTargetClasses',
            input_shape=(DEFAULTS['samples'], DEFAULTS['features']),
            target_classes=DEFAULTS['target_classes']
        )
        FuzzySelfOrganizer(
            name='FeaturesAndTargetClasses',
            features=DEFAULTS['features'],
            target_classes=DEFAULTS['target_classes']
        )

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='MaxNeuronsLessThanInitialNeurons',
                max_neurons=DEFAULTS['neurons'] - 1,
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='IfPartThresholdIsNegative',
                ifpart_threshold=-1,
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='IfPartSamplesIsNegative',
                ifpart_samples=-1,
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='ErrorDeltaIsNegative',
                err_delta=-1,
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='KSigLessThan1',
                k_sig=0
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='MaximumWidensIsNegative',
                max_widens=-1
            ))

        for prune_tol in [-1, 2]:
            with self.assertRaises(ValueError):
                FuzzySelfOrganizer(**_params(
                    name='PruneToleranceNotBetweenZeroAndOne',
                    prune_tol=prune_tol
                ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='KRMSELessThanOne',
                k_rmse=0
            ))

    def test_error_criterion(self):
        _, X_test, _, y_test = _get_training_data()

        sofnn = FuzzySelfOrganizer(model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'))
        y_pred = sofnn.model.predict(X_test)
        self.assertFalse(sofnn.error_criterion(y_pred, y_test))
        self.assertTrue(sofnn.error_criterion(y_pred, y_pred))

    def test_if_part_criterion(self):
        _, X_test, _, _ = _get_training_data()

        self.assertTrue(
            FuzzySelfOrganizer(
                model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
            ).if_part_criterion(X_test)
        )

    def test_min_dist_vector(self):
        _, X_test, _, _ = _get_training_data()

        min_dist = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
        ).minimum_distance_vector(X_test)

        self.assertTrue(
            numpy.allclose(
                min_dist,
                numpy.array([
                    [2.22789976, 2.22789976, 2.22789976],
                    [1.82063121, 1.82063121, 1.82063121],
                    [1.74726384, 1.74726384, 1.74726384],
                    [2.81454288, 2.81454288, 2.81454288]
                ])
            )
        )

    def test_duplicated_model(self):
        _, X_test, _, _ = _get_training_data()

        loaded_model = keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
        duplicated_model = FuzzySelfOrganizer(
            model=loaded_model
        ).duplicate_model()

        self.assertTrue(
            numpy.allclose(
                loaded_model.predict(X_test),
                duplicated_model.predict(X_test)
            )
        )

    def test_widening_centers(self):
        _, X_test, _, _ = _get_training_data()

        # if-part criterion already satisfied and weights unchanged when widening
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
        )
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertTrue(sofnn.if_part_criterion(X_test))
        self.assertTrue(sofnn.widen_centers(X_test))
        self.assertTrue(np.allclose(starting_weights, sofnn.model.fuzz.get_weights()))

        # do no widening iterations even when if-part criterion not satisfied
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'),
            max_widens=0
        )
        sofnn.model.fuzz.set_weights(
            [numpy.zeros_like(weight) for weight in sofnn.model.fuzz.get_weights()]
        )
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertFalse(sofnn.if_part_criterion(X_test))
        self.assertFalse(sofnn.widen_centers(X_test))
        self.assertTrue(np.allclose(starting_weights, sofnn.model.fuzz.get_weights()))

        # widen centers, but terminate before if-part criterion satisfied
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'),
            max_widens=5
        )
        sofnn.model.fuzz.set_weights(
            [numpy.ones_like(weight) for weight in sofnn.model.fuzz.get_weights()]
        )
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertFalse(sofnn.if_part_criterion(X_test))
        self.assertFalse(sofnn.widen_centers(X_test))
        self.assertFalse(np.allclose(starting_weights, sofnn.model.fuzz.get_weights()))
        self.assertTrue(
            np.allclose(
                sofnn.model.fuzz.get_weights(),
                [
                    numpy.array([[1., 1., 1.],
                                 [1., 1., 1.],
                                 [1., 1., 1.],
                                 [1., 1., 1.]]),
                    numpy.array([[1.2544, 1., 1.],
                                 [1.12, 1., 1.],
                                 [1.12, 1., 1.],
                                 [1.12, 1., 1.]])
                 ]
            )
        )

        # widen centers until if-part criterion satisfied
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
        )
        sofnn.model.fuzz.set_weights(
            [numpy.ones_like(weight) for weight in sofnn.model.fuzz.get_weights()]
        )
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertFalse(sofnn.if_part_criterion(X_test))
        self.assertTrue(sofnn.widen_centers(X_test))
        self.assertFalse(np.allclose(starting_weights, sofnn.model.fuzz.get_weights()))
        self.assertTrue(
            np.allclose(
                sofnn.model.fuzz.get_weights(),
                [
                    numpy.array([[1., 1., 1.],
                                 [1., 1., 1.],
                                 [1., 1., 1.],
                                 [1., 1., 1.]]),
                    numpy.array([[4.3634925, 1., 1.],
                                 [4.3634925, 1., 1.],
                                 [3.8959754, 1., 1.],
                                 [3.8959754, 1., 1.]])
                 ]
            )
        )

    def test_organize(self):
        X_train, X_test, y_train, y_test = _get_training_data()

        # CASE 1 - NO STRUCTURAL ADJUSTMENT. TRAIN MODEL
        # ERROR -> GOOD
        # IF-PART -> GOOD
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification-deep_trained.keras')
        )
        self.assertTrue(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        sofnn.organize(x=X_test, y=y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons)




        # sofnn.model.fuzz.set_weights(
        #     [numpy.ones_like(weight) for weight in sofnn.model.fuzz.get_weights()]
        # )
        # starting_weights = sofnn.model.fuzz.get_weights()
        # sofnn.organize(X_test)
        # self.assertFalse(np.allclose(starting_weights, sofnn.model.fuzz.get_weights()))
        # self.assertTrue(
        #     np.allclose(
        #         sofnn.model.fuzz.get_weights(),
        #         [
        #             numpy.array([[1., 1., 1.],
        #                          [1., 1., 1.],
        #                          [1., 1., 1.],
        #                          [1., 1., 1.]]),
        #             numpy.array([[4.3634925, 1., 1.],
        #                          [4.3634925, 1., 1.],
        #                          [3.8959754, 1., 1.],
        #                          [3.8959754, 1., 1.]])
        #         ]
        #     )
        # )




    # def test_init_with_features_and_input_shape(self):
    #     FuzzyNetwork(**_params(
    #         name='Only features provided.',
    #         features=DEFAULTS['features']
    #     ))
    #
    #     params = _params(
    #         name='Only input shape provided.',
    #         input_shape=(DEFAULTS['features'],)
    #     )
    #     params.pop('features')
    #     FuzzyNetwork(**params)
    #
    #     FuzzyNetwork(**_params(
    #         name='Input shape and feature agree.',
    #         features=DEFAULTS['features'],
    #         input_shape=(DEFAULTS['features'],)
    #     ))
    #
    #     FuzzyNetwork(**_params(
    #         name='Input shape and features parameters agree, and samples placeholder added to input shape.',
    #         features=DEFAULTS['features'],
    #         input_shape=(None, DEFAULTS['features'])
    #     ))
    #
    #     with self.assertRaises(ValueError):
    #         params = _params(name='Neither input shape or features are provided.')
    #         params.pop('features')
    #         FuzzyNetwork(**params)
    #
    #     with self.assertRaises(ValueError):
    #         FuzzyNetwork(**_params(
    #             name="Input shape and features don't agree.",
    #             features=DEFAULTS['features'],
    #             input_shape=(DEFAULTS['features'] + 1,)
    #         ))
    #
    #     with self.assertRaises(ValueError):
    #         FuzzyNetwork(**_params(
    #             name='Features < 1.',
    #             features=0,
    #         ))
    #
    # def test_basic_flow(self):
    #     model = FuzzyNetwork(
    #         features=DEFAULTS['features'],
    #         neurons=DEFAULTS['neurons'],
    #         problem_type=DEFAULTS['problem_type'],
    #         target_classes=DEFAULTS['target_classes']
    #     )
    #     self.assertEqual(len(model.layers), 5)
    #     # until the model sees example data, then the model will not be built and weights will not be added
    #     self.assertFalse(model.built)
    #     self.assertEqual(len(model.weights), 0)
    #
    #     # Test eager call
    #     x = numpy.random.random((DEFAULTS['samples'], DEFAULTS['features']))
    #     y = model(x)
    #     self.assertEqual(type(model), FuzzyNetwork)
    #     self.assertEqual(y.shape, (DEFAULTS['samples'], DEFAULTS['target_classes']))
    #
    #     # Test symbolic call
    #     x = K.KerasTensor((DEFAULTS['samples'], DEFAULTS['features']))
    #     y = model(x)
    #     self.assertEqual(y.shape, (DEFAULTS['samples'], DEFAULTS['target_classes']))
    #
    # def test_serialization(self):
    #     model = FuzzyNetwork(**_params(name='Serialization test.'))
    #     revived = self.run_class_serialization_test(model)
    #     self.assertLen(revived.layers, 5)
    #
    # def test_functional_properties(self):
    #     model = FuzzyNetwork(**_params(name='Functional properties test.'))
    #
    #     # self.assertEqual(model.inputs, INPUT DATA)
    #     #self.assertEqual(model.inputs, [FEATURES, NEURONS])
    #     #self.assertEqual(model.outputs, [model.layers[-1].output])
    #     self.assertEqual(model.input_shape, (None, DEFAULTS['features']))
    #     self.assertEqual(model.output_shape, (None, DEFAULTS['target_classes']))
    #
    # def test_pickleable(self):
    #     model = FuzzyNetwork(**_params(name='Pickleable test.'))
    #     result = pickle.loads(pickle.dumps(model))
    #     assert len(result.layers) == 5
    #
    # def test_hasattr(self):
    #     model = FuzzyNetwork(**_params(name='Attribute test.'))
    #     # TODO: add falses to check before/after model is compiled/fitted
    #     self.assertTrue(hasattr(model, "features"))
    #     self.assertTrue(hasattr(model, "input_shape"))
    #     self.assertTrue(hasattr(model, "output_shape"))
    #     self.assertTrue(hasattr(model, "neurons"))
    #     self.assertTrue(hasattr(model, "problem_type"))
    #     self.assertTrue(hasattr(model, "target_classes"))
    #     self.assertTrue(hasattr(model, "inputs"))
    #
    # def test_compile(self):
    #     model = FuzzyNetwork(**_params(
    #         name='Compile as classification.',
    #         problem_type='classification'
    #     ))
    #     model.compile()
    #     self.assertFalse(model.built)
    #
    #     model = FuzzyNetwork(**_params(
    #         name='Compile as regression.',
    #         problem_type='regression'
    #     ))
    #     model.compile()
    #     self.assertFalse(model.built)
    #
    # def test_fit_classification(self):
    #     epochs = 10
    #     X_train, X_test, y_train, y_test = _get_training_data()
    #
    #     trained_model = FuzzyNetwork(**_params(name='ModelFitTest'))
    #     trained_model.compile()
    #     trained_model.fit(X_train, y_train, epochs=epochs)
    #     #trained_model.save_weights(DATA_DIR / 'weights/classification.weights.h5')
    #
    #     loaded_model = FuzzyNetwork(**_params(name='LoadedModel'))
    #     loaded_model.compile()
    #     loaded_model.fit(X_train, y_train, epochs=1)
    #     self.assertTrue(loaded_model.built)
    #     loaded_model.load_weights(DATA_DIR / 'weights/classification.weights.h5')
    #
    #     assert np.allclose(trained_model.predict(X_test), loaded_model.predict(X_test))
    #
    # def test_fit_callbacks(self):
    #     epochs = 1
    #     X_train, X_test, y_train, y_test = _get_training_data()
    #
    #     model = FuzzyNetwork(**_params(name='AppendToOtherCallbacks'))
    #     model.compile()
    #     model.fit(X_train, y_train, epochs=epochs, callbacks=[
    #         ProgbarLogger()
    #     ])
    #
    #     model = FuzzyNetwork(**_params(name='InitializerCallbackAlreadyProvided'))
    #     model.compile()
    #     model.fit(X_train, y_train, epochs=epochs, callbacks=[
    #         FuzzyWeightsInitializer(
    #             sample_data=X_train,
    #             random_sample=False
    #         )
    #     ])
    #
    # def test_summary(self):
    #     model = FuzzyNetwork(**_params(name='Summary test.'))
    #     self.assertFalse(model.built)
    #     model.summary()
    #     self.assertFalse(model.built)
