import copy
import logging
from typing import Optional

from keras.api.layers import Input, Dense
from keras.api.metrics import CategoricalAccuracy, MeanSquaredError, Accuracy
from keras.api.models import Model
from keras.api.optimizers import Adam, RMSprop

from sofenn.callbacks import FuzzyWeightsInitializer
from sofenn.layers import FuzzyLayer, NormalizeLayer, WeightedLayer, OutputLayer
from sofenn.losses import CustomLoss

logger = logging.getLogger(__name__)


class FuzzyNetwork(Model):
    """
    Fuzzy Network
    =============
    Neural network using fuzzy logic modeling.

    -Implemented per description in:
        "An on-line algorithm for creating self-organizing
        fuzzy neural networks" - Leng, Prasad, McGinnity (2004)
    -Composed of 5 layers with varying "fuzzy rule" nodes (neurons).

    :param input_shape: Shape of input tensor.
    :param features: Number of features for input data.
    :param neurons: Number of neurons to use. Each one represents fuzzy neuron (if/then) operator.
    :param problem_type: Either 'classification' or 'regression' problem.
    :param target_classes: Optional number of classes in target data.
    :param name: Name of network (default: 'FuzzyNetwork').
    """
    def __init__(
            self,
            input_shape: Optional[tuple] = None,
            features: Optional[int] = None,
            neurons: int = 1,
            problem_type: str = 'classification',
            target_classes: Optional[int] = 1,
            name: str = 'FuzzyNetwork',
            **kwargs
    ):
        if features is not None and features < 1:
            raise ValueError('At least 1 input feature required')
        if features is not None and input_shape is not None:
            if input_shape not in [(features,), (None, features)]:
                raise ValueError('Input shape must match feature shape if providing both')
            self.features = features
        elif features is not None and input_shape is None:
            self.features = features
        elif features is None and input_shape is not None:
            self.features = input_shape[-1]
        else:
            raise ValueError('Must provide either features or input_shape')

        if neurons < 1:
            raise ValueError('Neurons must be a positive integer')
        self.neurons = neurons

        if problem_type.lower() not in ['classification', 'regression']:
            raise ValueError(f'Invalid problem type provided: {problem_type}')
        self.problem_type = problem_type.lower()

        if self.problem_type == 'classification':
            if target_classes is None:
                raise ValueError("Must provide target_classes parameter if 'problem_type' is 'classification'")
            elif target_classes < 2:
                raise ValueError("Must specify more than 1 target class if 'problem_type' is 'classification'")
        self.target_classes = None if self.problem_type == 'regression' else target_classes

        kwargs['name'] = kwargs.get('name', name)

        self.inputs = Input(name='Inputs', shape=(self.features,))
        self.fuzz = FuzzyLayer(shape=(self.features,), neurons=self.neurons)
        self.norm = NormalizeLayer(shape=(self.features, self.neurons))
        self.w = WeightedLayer(shape=[(self.features,), (self.neurons,)])
        self.raw = OutputLayer()
        self.softmax = Dense(self.target_classes, name='Softmax', activation='softmax')

        self.trained = False

        super().__init__(**kwargs)

    @property
    def input_shape(self) -> tuple:
        return None, self.features

    @property
    def output_shape(self) -> tuple:
        return None, 1 if self.target_classes is None else self.target_classes

    def call(self, inputs):
        """
        Call model.
        Note: keras.Model subclasses are not built until executing '.call()',
        as the model requires visibility into the input data to properly initialize.

        Layers
        ======
        0 - Input Layer
                input dataset
            - input shape  : (*, features)
        1 - Radial Basis Function Layer (Fuzzy Layer)
                layer to hold fuzzy rules for complex system
            - input : x
                shape: (*, features)
            - output : phi
                shape : (*, neurons)
        2 - Normalize Layer
                normalize each output of previous layer as
                relative amount from sum of all previous outputs
            - input : phi
                shape  : (*, neurons)
            - output : psi
                shape : (*, neurons)
        3 - Weighted Layer
                multiply bias vector (1+n_features, neurons) by
                parameter vector (1+n_features,) of parameters
                from each fuzzy rule
                multiply each product by output of each rule's
                layer from normalize layer
            - inputs : [x, psi]
                shape  : [(*, 1+features), (*, neurons)]
            - output : f
                shape : (*, neurons)
        4 - Output Layer
                summation of incoming signals from weighted layer
            - input shape  : (*, neurons)
            - output shape : (*,)

        5 - Softmax Layer (classification)
            softmax layer for classification problems
            - input shape : (*, 1)
            - output shape : (*, classes)

        """
        phi = self.fuzz(inputs)
        psi = self.norm(phi)
        f = self.w([inputs, psi])
        raw_output = self.raw(f)
        final_out = raw_output

        # add softmax layer for a classification problem
        if self.problem_type == 'classification':
            classes = self.softmax(raw_output)
            final_out = classes

        # TODO: determine logic for activation w/ regression
        # # extract activation from kwargs
        # if 'activation' not in kwargs:
        #     activation = 'sigmoid'
        # else:
        #     activation = kwargs['activation']
        # preds = Activation(name='OutputActivation', activation=activation)(raw_output)

        return final_out

    # TODO: add get_compile_config and compile_from_config(config)
    def compile(self, **kwargs) -> None:
        """Compile fuzzy network."""
        for key, default_value in self.compile_defaults(self.problem_type).items():
            if key not in kwargs:
                kwargs[key] = default_value
        super().compile(**kwargs)

    @staticmethod
    def compile_defaults(problem_type):
        """Generate kwargs dictionary with initialized defaults based on the problem type."""
        defaults = {
            'classification': {
                'loss': CustomLoss,
                'optimizer': Adam,
                'metrics': [CategoricalAccuracy]
            },
            'regression': {
                'loss': MeanSquaredError,
                'optimizer': RMSprop,
                'metrics': [Accuracy]
            }
        }
        return {
            key: [v() for v in val] if isinstance(val, list) else val()
            for key, val in copy.deepcopy(defaults[problem_type]).items()
        }

    def fit(self, *args, **kwargs):
        """Fit fuzzy network to training data."""
        if not self.built:
            logger.debug('FuzzyNetwork cannot be built until seeing training data')

        # add callback to instantiate fuzzy weights unless already provided
        x = kwargs['x'] if 'x' in kwargs else args[0]
        if 'callbacks' in kwargs:
            if any([isinstance(cb, FuzzyWeightsInitializer) for cb in kwargs['callbacks']]):
                logger.warning('User already provided Fuzzy Weight Initializer callback. '
                               'Will use existing Fuzzy Weight Initializer in kwargs')
            else:
                kwargs['callbacks'].append(FuzzyWeightsInitializer(sample_data=x))
        else:
            kwargs['callbacks'] = [FuzzyWeightsInitializer(sample_data=x)]

        super().fit(*args, **kwargs)
        if not self.trained:
            self.trained = True

    def summary(self, *args, **kwargs):
        """Show summary of fuzzy network."""
        x = Input(shape=(self.features,), name="InputRow")
        model = Model(inputs=[x], outputs=self.call(x), name=self.name + ' Summary')
        return model.summary(*args, **kwargs)

    def get_config(self):
        """Generate model config."""
        base_config = super().get_config()
        base_config['features'] = self.features
        base_config['neurons'] = self.neurons
        base_config['problem_type'] = self.problem_type
        base_config['target_classes'] = self.target_classes
        return base_config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return FuzzyNetwork(**config)
