import copy
from functools import lru_cache
from pathlib import Path

import keras
import numpy
import pandas
import pytest
from keras.src import testing
from sklearn.model_selection import train_test_split

from sofenn import FuzzyNetwork, FuzzySelfOrganizer

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

def _classification_model(deep=False):
    if deep:
        return keras.saving.load_model(DATA_DIR / 'models/iris_classification-deep.keras', custom_objects={'FuzzyNetwork': FuzzyNetwork})
    else:
        return keras.saving.load_model(DATA_DIR / 'models/iris_classification.keras', custom_objects={'FuzzyNetwork': FuzzyNetwork})


@pytest.mark.requires_trainable_backend
class FuzzySelfOrganizerTest(testing.TestCase):

    def test_init_with_model(self):
        model = FuzzyNetwork(**_params(
            name='Preinitialized model'
        ))
        assert model.get_config() == FuzzySelfOrganizer(model).model.get_config()

        FuzzySelfOrganizer(
            name='Input shape and target classes',
            input_shape=(DEFAULTS['samples'], DEFAULTS['features']),
            target_classes=DEFAULTS['target_classes']
        )
        FuzzySelfOrganizer(
            name='Features and target classes',
            features=DEFAULTS['features'],
            target_classes=DEFAULTS['target_classes']
        )

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='Max loops < 0',
                max_loops=-1,
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='Max neurons < initial neurons',
                max_neurons=DEFAULTS['neurons'] - 1,
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='If-part threshold < 0',
                ifpart_threshold=-1,
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='If-part samples < 0',
                ifpart_samples=-1,
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='Error delta < 0',
                error_delta=-5,
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='Widening factor < 0',
                k_sigma=-1
            ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='Maximum widens < 0',
                max_widens=-1
            ))

        for prune_tol in [-1, 2]:
            with self.assertRaises(ValueError):
                FuzzySelfOrganizer(**_params(
                    name='Prune tolerance not [0,1]',
                    prune_tol=prune_tol
                ))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(**_params(
                name='K root mean squared error < 1',
                k_rmse=0
            ))

    def test_error_criterion(self):
        _, X_test, _, y_test = _get_training_data()

        sofnn = FuzzySelfOrganizer(model=_classification_model())
        y_pred = sofnn.model.predict(X_test)
        self.assertFalse(sofnn.error_criterion(y_pred, y_test))
        self.assertTrue(sofnn.error_criterion(y_pred, y_pred))

    def test_if_part_criterion(self):
        _, X_test, _, _ = _get_training_data()

        self.assertTrue(
            FuzzySelfOrganizer(
                model=_classification_model()
            ).if_part_criterion(X_test)
        )

    def test_min_dist_vector(self):
        _, X_test, _, _ = _get_training_data()

        minimum_distance = FuzzySelfOrganizer(
            model=_classification_model()
        ).minimum_distance_vector(X_test)

        self.assertTrue(
            numpy.allclose(
                minimum_distance,
                numpy.array([
                    [2.22789976, 2.22789976, 2.22789976],
                    [1.82063121, 1.82063121, 1.82063121],
                    [1.74726384, 1.74726384, 1.74726384],
                    [2.81454288, 2.81454288, 2.81454288]
                ])
            )
        )

    def test_widening_centers(self):
        _, X_test, _, _ = _get_training_data()

        sofnn = FuzzySelfOrganizer(
            name='If-part criterion already satisfied and weights unchanged when widening',
            model=_classification_model()
        )
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertTrue(sofnn.if_part_criterion(X_test))
        self.assertTrue(sofnn.widen_centers(X_test))
        self.assertTrue(numpy.allclose(starting_weights, sofnn.model.fuzz.get_weights()))

        sofnn = FuzzySelfOrganizer(
            name='Do no widening iterations, even when the if-part criterion not satisfied',
            model=_classification_model(),
            max_widens=0
        )
        sofnn.model.fuzz.set_weights(
            [numpy.zeros_like(weight) for weight in sofnn.model.fuzz.get_weights()]
        )
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertFalse(sofnn.if_part_criterion(X_test))
        self.assertFalse(sofnn.widen_centers(X_test))
        self.assertTrue(numpy.allclose(starting_weights, sofnn.model.fuzz.get_weights()))

        sofnn = FuzzySelfOrganizer(
            name='Widen centers, but terminate before the if-part criterion satisfied',
            model=_classification_model(),
            max_widens=5
        )
        sofnn.model.fuzz.set_weights(
            [numpy.ones_like(weight) for weight in sofnn.model.fuzz.get_weights()]
        )
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertFalse(sofnn.if_part_criterion(X_test))
        self.assertFalse(sofnn.widen_centers(X_test))
        self.assertFalse(numpy.allclose(starting_weights, sofnn.model.fuzz.get_weights()))
        self.assertTrue(
            numpy.allclose(
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

        sofnn = FuzzySelfOrganizer(
            name='Widen centers until the if-part criterion satisfied',
            model=_classification_model()
        )
        sofnn.model.fuzz.set_weights(
            [numpy.ones_like(weight) for weight in sofnn.model.fuzz.get_weights()]
        )
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertFalse(sofnn.if_part_criterion(X_test))
        self.assertTrue(sofnn.widen_centers(X_test))
        self.assertFalse(numpy.allclose(starting_weights, sofnn.model.fuzz.get_weights()))
        self.assertTrue(
            numpy.allclose(
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
            model=_classification_model()
        )
        starting_neurons = sofnn.model.neurons
        self.assertTrue(sofnn.add_neuron(X_train, y_train))
        self.assertTrue(sofnn.model.neurons == starting_neurons + 1)

    def test_new_neuron_weights(self):
        X_train, _, _, _ = _get_training_data()

        sofnn = FuzzySelfOrganizer(
            model=_classification_model()
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
            model=_classification_model()
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
                name='One neuron',
                neurons=1
            ))
        )
        self.assertFalse(sofnn.prune_neurons(X_train, y_train))

        sofnn = FuzzySelfOrganizer(
            name='Starting neurons greater than or equal to initial neurons',
            model=_classification_model()
        )
        sofnn.model.compile()
        starting_neurons = sofnn.model.neurons
        self.assertFalse(sofnn.prune_neurons(X_train, y_train))
        self.assertTrue(starting_neurons >= sofnn.model.neurons)

        sofnn = FuzzySelfOrganizer(
            name='Only one neuron above the prune threshold',
            model=_classification_model(),
            prune_threshold=0.99,
            k_rmse=0.4390
        )
        sofnn.model.compile()
        starting_neurons = sofnn.model.neurons
        self.assertTrue(sofnn.prune_neurons(X_train, y_train))
        self.assertTrue(sofnn.model.neurons == starting_neurons - 1)

        sofnn = FuzzySelfOrganizer(
            name='Prune all but last neuron',
            model=_classification_model(),
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

        sofnn = FuzzySelfOrganizer(
            name='No structural adjustment '
                 'Error: Pass '
                 'If-Part: Pass',
            model=_classification_model(deep=True),
            error_delta=0.15
        )
        self.assertTrue(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        sofnn.organize(X_test, y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons)

        sofnn = FuzzySelfOrganizer(
            name='Widen centers '
                 'Error: Pass '
                 'If-Part: Fail',
            model=_classification_model(deep=True),
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
        sofnn.organize(X_test, y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons)
        final_weights = sofnn.model.get_weights()
        self.assertFalse(numpy.allclose(starting_weights[1], final_weights[1])) # confirm center weights are different

        sofnn = FuzzySelfOrganizer(
            name='Add neuron and retrain model '
                 'Error: Fail '
                 'If-Part: Pass',
            model=_classification_model()
        )
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        sofnn.organize(X_test, y_test, epochs=1)
        self.assertTrue(sofnn.model.neurons == starting_neurons + 1)

        sofnn = FuzzySelfOrganizer(
            name='Widen centers and no need to add neuron '
                 'Error: Fail '
                 'If-Part: Fail',
            model=_classification_model(),
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
        sofnn.organize(X_test, y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons)
        final_weights = sofnn.model.get_weights()
        self.assertFalse(numpy.allclose(starting_weights[1], final_weights[1])) # confirm center weights are different
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))

        sofnn = FuzzySelfOrganizer(
            name='Add neuron after widening centers fails '
                 'Error: Fail '
                 'If-Part: Fail',
            model=_classification_model(),
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
        sofnn.organize(X_test, y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons + 1)
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))

        sofnn = FuzzySelfOrganizer(
            name='Prune neuron '
                 'Error: Fail '
                 'If-Part: Fail',
            model=_classification_model(),
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
            name='Fail to organize',
            model=_classification_model(),
            max_loops=5,
            epochs=1
        )
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        self.assertFalse(sofnn.self_organize(X_test, y_test, epochs=1))
        self.assertTrue(sofnn.model.neurons > starting_neurons)

        sofnn = FuzzySelfOrganizer(
            name='Stop at max neurons',
            model=_classification_model(),
            max_loops=5,
            max_neurons=3,
            epochs=1
        )
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile()
        self.assertFalse(sofnn.self_organize(X_test, y_test, epochs=1))
        self.assertTrue(sofnn.model.neurons > starting_neurons)

        sofnn = FuzzySelfOrganizer(
            name='Successfully organize',
            model=_classification_model(deep=True),
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
