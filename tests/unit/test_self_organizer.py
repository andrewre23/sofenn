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
                name='MaxLoopsLessThanZero',
                max_loops=-1,
            ))

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

    def test_add_neuron(self):
        X_train, _, y_train, _ = _get_training_data()

        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
        )
        starting_neurons = sofnn.model.neurons
        self.assertTrue(sofnn.add_neuron(X_train, y_train))
        self.assertTrue(sofnn.model.neurons == starting_neurons + 1)

    def test_new_neuron_weights(self):
        X_train, _, _, _ = _get_training_data()

        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
        )
        ck, sk = sofnn.new_neuron_weights(X_train)
        self.assertTrue(
            numpy.allclose(
                ck,
                numpy.array([5.80018616, 2.62788224, 3.9982748 , 1.21259259]))
        )
        self.assertTrue(
            numpy.allclose(
                sk,
                numpy.array([4.00925207, 4.0355587 , 4.02074146, 2.86290745])
            )
        )

    def test_rebuild_model(self):
        X_train, _, y_train, _ = _get_training_data()

        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
        )

        rebuilt = sofnn.rebuild_model(X_train, y_train, sofnn.model.neurons, sofnn.model.get_weights())
        self.assertTrue(sofnn.model.neurons == rebuilt.neurons)
        for i, original_weight in enumerate(sofnn.model.fuzz.get_weights()):
            self.assertTrue(
                numpy.allclose(
                    original_weight,
                    rebuilt.weights[i],
                )
            )

    def test_prune_neuron(self):
        X_train, _, y_train, _ = _get_training_data()

        sofnn = FuzzySelfOrganizer(
            model=FuzzyNetwork(**_params(
                name='OneNeuron',
                neurons=1
            ))
        )
        self.assertFalse(sofnn.prune_neurons(X_train, y_train))

        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
        )
        sofnn.model.compile()
        starting_neurons = sofnn.model.neurons
        self.assertFalse(sofnn.prune_neurons(X_train, y_train))
        self.assertTrue(starting_neurons >= sofnn.model.neurons)

        # only 1 neuron above threshold to prune
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'),
            prune_threshold=0.99,
            k_rmse=0.4390
        )
        sofnn.model.compile()
        starting_neurons = sofnn.model.neurons
        self.assertTrue(sofnn.prune_neurons(X_train, y_train))
        self.assertTrue(sofnn.model.neurons == starting_neurons - 1)

        # prune all but last neuron
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'),
            prune_threshold=0.99,
            k_rmse=5
        )
        sofnn.model.compile()
        starting_neurons = sofnn.model.neurons
        self.assertTrue(sofnn.prune_neurons(X_train, y_train))
        self.assertTrue(sofnn.model.neurons == starting_neurons - 2 == 1)

    def test_combine_membership_functions(self):
        with self.assertRaises(NotImplementedError):
            FuzzySelfOrganizer(model=FuzzyNetwork(**_params())).combine_membership_functions()

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

        # CASE 2 - WIDEN CENTERS
        # ERROR -> GOOD
        # IF-PART -> BAD
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification-deep_trained.keras'),
            ifpart_threshold=0.9,
            ifpart_samples=0.99,
            error_delta=0.5,
            max_widens=100,
        )
        self.assertTrue(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        starting_weights = sofnn.model.get_weights()
        sofnn.model.compile()
        sofnn.organize(x=X_test, y=y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons)
        final_weights = sofnn.model.get_weights()
        self.assertFalse(np.allclose(starting_weights[1], final_weights[1])) # confirm center weights are different

        # CASE 3 - ADD NEURON. RE-TRAIN MODEL
        # ERROR -> BAD
        # IF-PART -> GOOD
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras')
        )
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        sofnn.organize(x=X_test, y=y_test, epochs=1)
        self.assertTrue(sofnn.model.neurons == starting_neurons + 1)

        # CASE 4A - WIDEN CENTERS AND NO NEED TO ADD NEURON
        # ERROR -> BAD
        # IF-PART -> BAD
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'),
            ifpart_threshold=0.9,
            ifpart_samples=0.99,
            error_delta=0.1,
            max_widens=250
        )
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        starting_weights = sofnn.model.get_weights()
        sofnn.model.compile()
        sofnn.organize(x=X_test, y=y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons)
        final_weights = sofnn.model.get_weights()
        self.assertFalse(np.allclose(starting_weights[1], final_weights[1])) # confirm center weights are different
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))

        # CASE 4B - ADD NEURON AFTER WIDENING CENTERS FAILS
        # ERROR -> BAD
        # IF-PART -> BAD
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'),
            ifpart_threshold=0.9,
            ifpart_samples=0.99,
            error_delta=0.1,
            max_widens=0,
            epochs=100
        )
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        sofnn.organize(x=X_test, y=y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons + 1)
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))

        # CASE 5 - PRUNE NEURON
        # ERROR -> BAD
        # IF-PART -> BAD
        sofnn = FuzzySelfOrganizer(
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'),
            ifpart_threshold=0.9,
            ifpart_samples=0.99,
            error_delta=0.1,
            max_widens=0,
            prune_threshold=0.99,
            k_rmse=5,
            epochs=3
        )
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        sofnn.organize(X_test, y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons - 2 == 1)
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))

    def test_self_organize(self):
        X_train, X_test, y_train, y_test = _get_training_data()

        sofnn = FuzzySelfOrganizer(
            name='FailToOrganize',
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'),
            max_loops=5,
            epochs=1
        )
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        self.assertFalse(sofnn.self_organize(X_test, y_test, epochs=1))
        self.assertTrue(sofnn.model.neurons > starting_neurons)

        sofnn = FuzzySelfOrganizer(
            name='StopAtMaxNeurons',
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras'),
            max_loops=5,
            max_neurons=3,
            epochs=1
        )
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        self.assertFalse(sofnn.self_organize(X_test, y_test, epochs=1))
        self.assertTrue(sofnn.model.neurons > starting_neurons)

        sofnn = FuzzySelfOrganizer(
            name='SuccessfullyOrganize',
            model=keras.saving.load_model(DATA_DIR / 'models/iris_classification-deep_trained.keras'),
            max_loops=3,
            error_delta=0.99,
            epochs=5
        )
        self.assertTrue(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        self.assertTrue(sofnn.self_organize(X_test, y_test, epochs=5))
        self.assertTrue(sofnn.model.neurons == starting_neurons)
