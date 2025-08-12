import pytest
from absl.testing import parameterized
from keras.src import testing

from sofenn import FuzzyNetwork, FuzzySelfOrganizer
from tests.testing_utils import PROBLEM_TYPES, _init_params, _load_saved_model


@pytest.mark.requires_trainable_backend
class FuzzySelfOrganizerTest(testing.TestCase):

    @parameterized.named_parameters(PROBLEM_TYPES)
    def test_passing_model_to_organizer(self, problem_type):
        model_from_init = FuzzyNetwork(name='Fuzzy Network', **_init_params(problem_type))
        loaded_model = _load_saved_model(problem_type)

        for model in [model_from_init, loaded_model]:
            sofnn = FuzzySelfOrganizer(name='SelfOrganizer', model=model)

            self.assertTrue(model.features == sofnn.model.features)
            self.assertTrue(model.input_shape == sofnn.model.input_shape)
            self.assertTrue(model.neurons == sofnn.model.neurons)
            self.assertTrue(model.num_classes == sofnn.model.num_classes)
            self.assertTrue(model.inputs == sofnn.model.inputs)
