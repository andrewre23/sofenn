import inspect

import numpy
import pytest
from absl.testing import parameterized
from keras.src import testing

from sofenn import FuzzyNetwork, FuzzySelfOrganizer
from tests.testing_utils import PROBLEM_DEFAULTS, PROBLEM_TYPES, \
    _get_training_data, _load_saved_model, _init_params, _compile_params


@pytest.mark.requires_trainable_backend
class FuzzySelfOrganizerTest(testing.TestCase):

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_init_with_model(self, problem_type):

        model = FuzzyNetwork(name='Preinitialized model', **_init_params(problem_type))
        self.assertEqual(model.get_config(), FuzzySelfOrganizer(model).model.get_config())

        FuzzySelfOrganizer(name='Initialize model on self-organizer initialization', **_init_params(problem_type))

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_input_validation(self, problem_type):
        defaults = PROBLEM_DEFAULTS[problem_type]

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(name='Max loops < 0', **_init_params(problem_type, max_loops=-1))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(name='Max neurons < initial neurons', **_init_params(problem_type, max_neurons=defaults['neurons'] - 1))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(name='If-part threshold < 0', **_init_params(problem_type, ifpart_threshold=-1))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(name='If-part samples < 0', **_init_params(problem_type, ifpart_samples=-1))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(name='Error delta < 0', **_init_params(problem_type, error_delta=-5))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(name='Widening factor < 0', **_init_params(problem_type, k_sigma=-1))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(name='Maximum widens < 0', **_init_params(problem_type, max_widens=-1))

        for prune_tol in [-1, 2]:
            with self.assertRaises(ValueError):
                FuzzySelfOrganizer(name='Prune tolerance not [0,1]', **_init_params(problem_type, prune_tol=prune_tol))

        with self.assertRaises(ValueError):
            FuzzySelfOrganizer(name='K root mean squared error < 1', **_init_params(problem_type, k_rmse=0))

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_error_criterion(self, problem_type):
        _, X_test, _, y_test = _get_training_data(problem_type)

        sofnn = FuzzySelfOrganizer(model=_load_saved_model(problem_type))
        y_pred = sofnn.model.predict(X_test)
        self.assertFalse(sofnn.error_criterion(y_pred, y_test))
        self.assertTrue(sofnn.error_criterion(y_pred, y_pred))

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_if_part_criterion(self, problem_type):
        _, X_test, _, _ = _get_training_data(problem_type)

        self.assertTrue(FuzzySelfOrganizer(model=_load_saved_model(problem_type)).if_part_criterion(X_test))

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_min_dist_vector(self, problem_type):
        _, X_test, _, _ = _get_training_data(problem_type)

        target_distance = {
            'classification':
                numpy.array([
                    [[2.23177525, 2.23177525, 2.23177525, 2.23177525, 2.23177525],
                     [1.75618350, 1.75618350, 1.75618360, 1.75618360, 1.75618360],
                     [1.78014647, 1.78014647, 1.78014647, 1.78014647,1.780146468],
                     [2.75448100, 2.75448110, 2.75448110, 2.75448110, 2.75448110]]
                ]),
            'regression':
                numpy.array([
                    [1.16613078, 1.16613078, 1.166130785],
                    [1.17406269, 1.17406269, 1.503086111],
                    [1.47902240, 1.47902240, 1.479022395],
                    [1.50177162, 1.61061059, 1.610610588],
                    [1.51192624, 1.51192624, 1.511926239]
                ])
        }

        minimum_distance = FuzzySelfOrganizer(model=_load_saved_model(problem_type)).minimum_distance_vector(X_test)

        self.assertTrue(
            numpy.allclose(
                minimum_distance,
                target_distance[problem_type]
            )
        )

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_widening_centers(self, problem_type):
        centers = {
            'failure': {
                'classification': {
                    'c': numpy.array([
                        [1., 1., 1., 1., 1.],
                        [1., 1., 1., 1., 1.],
                        [1., 1., 1., 1., 1.],
                        [1., 1., 1., 1., 1.]
                    ]),
                    's': numpy.array([
                        [1.25440001, 1., 1., 1., 1.],
                        [1.12      , 1., 1., 1., 1.],
                        [1.12      , 1., 1., 1., 1.],
                        [1.12      , 1., 1., 1., 1.]
                    ]),
                },
                'regression': {
                    'c': numpy.array([
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]
                    ]),
                    's': numpy.array([
                        [1.12, 1., 1.],
                        [1.12, 1., 1.],
                        [1.12, 1., 1.],
                        [1.12, 1., 1.],
                        [1.12, 1., 1.]
                    ]),
                },
            },
            'success': {
                'classification': {
                    'c': numpy.array([
                        [[1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1.],
                         [1., 1., 1., 1., 1.]]
                    ]),
                    's': numpy.array([
                        [4.36349344, 1., 1., 1., 1.],
                        [4.36349344, 1., 1., 1., 1.],
                        [3.89597607, 1., 1., 1., 1.],
                        [3.89597607, 1., 1., 1., 1.]]
                    ),
                },
                'regression': {
                    'c': numpy.array([
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]
                    ]),
                    's': numpy.array([
                        [[4.363493, 1., 1.],
                         [3.895976, 1., 1.],
                         [3.895976, 1., 1.],
                         [3.895976, 1., 1.],
                         [3.895976, 1., 1.]]
                    ]),
                },
            }
        }

        ifpart_threshold = .9 if problem_type == 'regression' else \
            inspect.signature(FuzzySelfOrganizer).parameters['ifpart_threshold'].default

        _, X_test, _, _ = _get_training_data(problem_type)

        sofnn = FuzzySelfOrganizer(
            name='If-part criterion already satisfied and weights unchanged when widening',
            model=_load_saved_model(problem_type)
        )
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertTrue(sofnn.if_part_criterion(X_test))
        self.assertTrue(sofnn.widen_centers(X_test))
        self.assertTrue(numpy.allclose(starting_weights, sofnn.model.fuzz.get_weights()))

        sofnn = FuzzySelfOrganizer(
            name='Do no widening iterations, even when the if-part criterion not satisfied',
            model=_load_saved_model(problem_type),
            max_widens=0
        )
        sofnn.model.fuzz.set_weights([numpy.zeros_like(weight) for weight in sofnn.model.fuzz.get_weights()])
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertFalse(sofnn.if_part_criterion(X_test))
        self.assertFalse(sofnn.widen_centers(X_test))
        self.assertTrue(numpy.allclose(starting_weights, sofnn.model.fuzz.get_weights()))

        sofnn = FuzzySelfOrganizer(
            name='Widen centers, but terminate before the if-part criterion satisfied',
            model=_load_saved_model(problem_type),
            max_widens=5,
            ifpart_threshold=ifpart_threshold
        )
        sofnn.model.fuzz.set_weights([numpy.ones_like(weight) for weight in sofnn.model.fuzz.get_weights()])
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertFalse(sofnn.if_part_criterion(X_test))
        self.assertFalse(sofnn.widen_centers(X_test))
        self.assertFalse(numpy.allclose(starting_weights, sofnn.model.fuzz.get_weights()))

        targets = centers['failure'][problem_type]
        for i, w in enumerate(['c', 's']):
            self.assertTrue(
                numpy.allclose(
                    sofnn.model.fuzz.get_weights()[i],
                    targets[w]
                )
            )

        sofnn = FuzzySelfOrganizer(
            name='Widen centers until the if-part criterion satisfied',
            model=_load_saved_model(problem_type),
            ifpart_threshold=ifpart_threshold
        )
        sofnn.model.fuzz.set_weights([numpy.ones_like(weight) for weight in sofnn.model.fuzz.get_weights()])
        starting_weights = sofnn.model.fuzz.get_weights()
        self.assertFalse(sofnn.if_part_criterion(X_test))
        self.assertTrue(sofnn.widen_centers(X_test))
        self.assertFalse(numpy.allclose(starting_weights, sofnn.model.fuzz.get_weights()))

        targets = centers['success'][problem_type]
        for i, w in enumerate(['c', 's']):
            self.assertTrue(
                numpy.allclose(
                    sofnn.model.fuzz.get_weights()[i],
                    targets[w]
                )
            )

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_add_neuron(self, problem_type):
        X_train, _, y_train, _ = _get_training_data(problem_type)

        sofnn = FuzzySelfOrganizer(model=_load_saved_model(problem_type))
        starting_neurons = sofnn.model.neurons
        self.assertTrue(sofnn.add_neuron(X_train, y_train, **_compile_params(problem_type)))
        self.assertTrue(sofnn.model.neurons == starting_neurons + 1)

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_new_neuron_weights(self, problem_type):
        X_train, _, _, _ = _get_training_data(problem_type)

        new_weights = {
            'classification': {
                'ck': [5.789541, 2.60152316, 3.97809458, 1.21259259],
                'sk': [3.998332, 4.00057650, 4.00002050, 2.87433151]
            },
            'regression': {
                'ck': [0.49599802, 0.46910115, 0.47057344, 0.536142225, 0.498671019],
                'sk': [1.45126167, 1.46830095, 1.45824282, 1.421873220, 1.416617054]
            },
        }

        sofnn = FuzzySelfOrganizer(model=_load_saved_model(problem_type))
        ck, sk = sofnn.new_neuron_weights(X_train)

        self.assertTrue(
            numpy.allclose(
                ck,
                new_weights[problem_type]['ck']
            )
        )
        self.assertTrue(
            numpy.allclose(
                sk,
                new_weights[problem_type]['sk']
            )
        )

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_rebuild_model(self, problem_type):
        X_train, _, y_train, _ = _get_training_data(problem_type)

        sofnn = FuzzySelfOrganizer(model=_load_saved_model(problem_type))

        rebuilt = sofnn.rebuild_model(
            X_train, y_train,
            sofnn.model.neurons,
            sofnn.model.get_weights(),
            **_compile_params(problem_type)
        )
        self.assertTrue(sofnn.model.neurons == rebuilt.neurons)
        for i, original_weight in enumerate(sofnn.model.fuzz.get_weights()):
            self.assertTrue(
                numpy.allclose(
                    original_weight,
                    rebuilt.weights[i],
                )
            )

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_prune_neuron(self, problem_type):

        X_train, _, y_train, _ = _get_training_data(problem_type)

        sofnn = FuzzySelfOrganizer(model=FuzzyNetwork(name='One neuron', **_init_params(problem_type, neurons=1)))
        self.assertFalse(sofnn.prune_neurons(X_train, y_train, **_compile_params(problem_type)))

        sofnn = FuzzySelfOrganizer(
            name='Starting neurons greater than or equal to initial neurons',
            model=_load_saved_model(problem_type)
        )
        sofnn.model.compile()
        starting_neurons = sofnn.model.neurons
        sofnn.prune_neurons(X_train, y_train, **_compile_params(problem_type))
        self.assertTrue(starting_neurons >= sofnn.model.neurons)

        sofnn = FuzzySelfOrganizer(
            name='Prune neurons',
            model=_load_saved_model(problem_type),
            prune_threshold=0.99,
            k_rmse=0.4390
        )
        # TODO: delete passing of compile params
        sofnn.model.compile(**_compile_params(problem_type))
        starting_neurons = sofnn.model.neurons
        self.assertTrue(sofnn.prune_neurons(X_train, y_train, **_compile_params(problem_type)))
        self.assertTrue(sofnn.model.neurons < starting_neurons)

        sofnn = FuzzySelfOrganizer(
            name='Prune all but last neuron',
            model=_load_saved_model(problem_type),
            prune_threshold=0.99,
            k_rmse=5
        )
        sofnn.model.compile(**_compile_params(problem_type))
        starting_neurons = sofnn.model.neurons
        self.assertTrue(sofnn.prune_neurons(X_train, y_train, **_compile_params(problem_type)))
        self.assertTrue(sofnn.model.neurons == 1 < starting_neurons)

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_combine_membership_functions(self, problem_type):
        with self.assertRaises(NotImplementedError):
            FuzzySelfOrganizer(model=FuzzyNetwork(**_init_params(problem_type))).combine_membership_functions()

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_organize(self, problem_type):
        X_train, X_test, y_train, y_test = _get_training_data(problem_type)

        name = 'No structural adjustment ' \
               'Error: Pass ' \
               'If-Part: Pass'
        params = {
            'classification': {
                'error_delta': 4,
                'prune_tol': 0.001,
            },
            'regression': {
                'error_delta': 4,
                'prune_tol': 0.001,
            }
        }
        sofnn = FuzzySelfOrganizer(
            name=name,
            model=_load_saved_model(problem_type),
            **params[problem_type]
        )
        self.assertTrue(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        # TODO: remove need to explicitly compile where appropriate. loading? training?
        sofnn.model.compile(**_compile_params(problem_type))
        sofnn.organize(X_test, y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons)

        name = 'Widen centers '\
               'Error: Pass ' \
               'If-Part: Fail'
        params = {
            'classification': {
                'ifpart_threshold': 0.9,
                'ifpart_samples': 0.99,
                'error_delta': 0.5,
                'max_widens': 100,
            },
            'regression': {
                'ifpart_threshold': 0.9,
                'ifpart_samples': 0.99,
                'error_delta': 4,
                'max_widens': 100,
                'prune_tol': 0.001,
            }
        }
        sofnn = FuzzySelfOrganizer(
            name=name,
            model=_load_saved_model(problem_type, deep=True),
            **params[problem_type]
        )
        self.assertTrue(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        starting_weights = sofnn.model.get_weights()
        sofnn.model.compile(**_compile_params(problem_type))
        sofnn.organize(X_test, y_test)
        self.assertTrue(sofnn.model.neurons == starting_neurons)
        final_weights = sofnn.model.get_weights()
        self.assertFalse(numpy.allclose(starting_weights[1], final_weights[1])) # confirm center weights are different

        name = 'Add neuron and retrain model ' \
               'Error: Fail ' \
               'If-Part: Pass'
        params = {
            'classification': {
                'ifpart_threshold': 0.5,
                'ifpart_samples': 0.5,
                'error_delta': 0.4,
                'max_widens': 100,
            },
            'regression': {
                'ifpart_threshold': 0.5,
                'ifpart_samples': 0.5,
                'error_delta': 0.5,
                'max_widens': 100,
                'prune_tol': 0.001,
                'k_rmse': 0.01
            }
        }
        sofnn = FuzzySelfOrganizer(
            name=name,
            model=_load_saved_model(problem_type),
            **params[problem_type]
        )
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile(**_compile_params(problem_type))
        sofnn.organize(X_test, y_test, **_compile_params(problem_type, epochs=1))
        self.assertTrue(sofnn.model.neurons == starting_neurons + 1)

        name = 'Widen centers and no need to add neuron ' \
               'Error: Fail ' \
               'If-Part: Fail'
        params = {
            'classification': {
                'ifpart_threshold': 0.9,
                'ifpart_samples': 0.99,
                'error_delta': 0.1,
                'max_widens': 250,
            },
            'regression': {
                'ifpart_threshold': 0.999,
                'ifpart_samples': 0.99,
                'error_delta': 0.5,
                'max_widens': 250,
                'prune_tol': 0.01
            }
        }
        sofnn = FuzzySelfOrganizer(
            name=name,
            model=_load_saved_model(problem_type),
            **params[problem_type]
        )
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        starting_weights = sofnn.model.get_weights()
        sofnn.model.compile(**_compile_params(problem_type))
        sofnn.organize(X_test, y_test, **_compile_params(problem_type))
        self.assertTrue(sofnn.model.neurons == starting_neurons)
        final_weights = sofnn.model.get_weights()
        self.assertFalse(numpy.allclose(starting_weights[1], final_weights[1])) # confirm center weights are different
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))

        name = 'Add neuron after widening centers fails ' \
               'Error: Fail ' \
               'If-Part: Fail'
        params = {
            'classification': {
                'ifpart_threshold': 0.99999,
                'ifpart_samples': 0.99,
                'error_delta': 0.1,
                'max_widens': 0,
                'epochs': 100
            },
            'regression': {
                'ifpart_threshold': 0.99999,
                'ifpart_samples': 0.99,
                'error_delta': 0.1,
                'max_widens': 0,
                'prune_tol': 0.01,
                'epochs': 100
            }
        }
        sofnn = FuzzySelfOrganizer(
            name=name,
            model=_load_saved_model(problem_type),
            **params[problem_type]
        )
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile(**_compile_params(problem_type))
        sofnn.organize(X_test, y_test, **_compile_params(problem_type))
        self.assertTrue(sofnn.model.neurons == starting_neurons + 1)
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))

        name = 'Prune neuron ' \
               'Error: Fail ' \
               'If-Part: Fail'
        params = {
            'classification': {
                'ifpart_threshold': 0.99999,
                'ifpart_samples': 0.99,
                'error_delta': 0.1,
                'max_widens': 0,
                'prune_threshold': 0.99,
                'k_rmse': 5,
                'epochs': 3
            },
            'regression': {
                'ifpart_threshold': 0.99999,
                'ifpart_samples': 0.99,
                'error_delta': 0.1,
                'max_widens': 0,
                'prune_threshold': 0.99,
                'k_rmse': 5,
                'epochs': 3
            }
        }
        sofnn = FuzzySelfOrganizer(
            name=name,
            model=_load_saved_model(problem_type),
            **params[problem_type]
        )
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile(**_compile_params(problem_type))
        sofnn.organize(X_test, y_test, **_compile_params(problem_type))
        self.assertTrue(sofnn.model.neurons == 1 < starting_neurons)
        self.assertFalse(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertFalse(sofnn.if_part_criterion(X_test))

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_self_organize(self, problem_type):

        X_train, X_test, y_train, y_test = _get_training_data(problem_type)

        name = 'Fail to organize'
        params = {
            'classification': {
                'max_loops': 5,
                'epochs': 1,
            },
            'regression': {
                'prune_tol': 0.01,
                'max_loops': 5,
                'epochs': 1,
            }
        }
        sofnn = FuzzySelfOrganizer(
            name=name,
            model=_load_saved_model(problem_type, deep=True),
            **params[problem_type]
        )
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile(**_compile_params(problem_type))
        self.assertFalse(sofnn.self_organize(X_test, y_test, **_compile_params(problem_type), epochs=1))
        self.assertTrue(sofnn.model.neurons > starting_neurons)

        name = 'Stop at max neurons'
        params = {
            'classification': {
                'max_loops': 5,
                'max_neurons': 7,
                'prune_tol': 0.01,
                'epochs': 1,
            },
            'regression': {
                'max_loops': 5,
                'max_neurons': 3,
                'prune_tol': 0.01,
                'epochs': 1,
            }
        }
        sofnn = FuzzySelfOrganizer(
            name=name,
            model=_load_saved_model(problem_type),
            **params[problem_type]
        )
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile(**_compile_params(problem_type))
        self.assertFalse(sofnn.self_organize(X_test, y_test, **_compile_params(problem_type), epochs=1))
        self.assertTrue(sofnn.model.neurons > starting_neurons)

        name = 'Successfully organize'
        params = {
            'classification': {
                'ifpart_threshold': 0.001,
                'ifpart_samples': 0.5,
                'error_delta': 4,
                'prune_tol': 0.001,
            },
            'regression': {
                'ifpart_threshold': 0.001,
                'ifpart_samples': 0.5,
                'error_delta': 4,
                'prune_tol': 0.001,
            }
        }
        sofnn = FuzzySelfOrganizer(
            name=name,
            model=_load_saved_model(problem_type, deep=True),
            **params[problem_type]
        )
        self.assertTrue(sofnn.error_criterion(y_test, sofnn.model.predict(X_test)))
        self.assertTrue(sofnn.if_part_criterion(X_test))
        starting_neurons = sofnn.model.neurons
        sofnn.model.compile(**_compile_params(problem_type))
        self.assertTrue(sofnn.self_organize(X_test, y_test, **_compile_params(problem_type), epochs=5))
        self.assertTrue(sofnn.model.neurons == starting_neurons)
