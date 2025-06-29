import logging
from typing import Optional

# TODO: delete from keras.src.<module> in favor of kears.<module> imports
import keras
from keras.activations import get as get_activation
from keras.activations import linear
from keras.activations import serialize
from keras.layers import Input
from keras.models import Model

from sofenn.callbacks import FuzzyWeightsInitializer
from sofenn.layers import FuzzyLayer, NormalizeLayer, WeightedLayer, OutputLayer
from sofenn.utils.layers import parse_function_kwargs

logger = logging.getLogger(__name__)

# TODO: test for minimum level of serializable needed
@keras.saving.register_keras_serializable(package='sofenn')
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
            # TODO: remove need for specifying features, and only take input_shape
            input_shape: tuple,
            neurons: int = 1,
            num_classes: int = 1,
            activation: Optional[str] = linear,
            name: str = 'FuzzyNetwork',
            **kwargs
    ):
        if neurons < 1:
            raise ValueError('Neurons must be a positive integer')
        self.neurons = neurons
        if num_classes < 1:
            raise ValueError('Number of classes must be a positive integer')
        self.num_classes = num_classes

        kwargs['name'] = kwargs.get('name', name)
        self.inputs = Input(name='Inputs', shape=input_shape)
        self.fuzz = FuzzyLayer(input_shape=input_shape, neurons=neurons)
        self.norm = NormalizeLayer(input_shape=input_shape)
        self.w = WeightedLayer(input_shape=[input_shape, (neurons,)])
        self.final_output = OutputLayer(input_shape=input_shape, num_classes=num_classes, activation=activation)
        self.input_shape = input_shape

        self.trained = False

        super().__init__(**kwargs)

    @property
    def features(self) -> tuple:
        return self.input_shape[-1]

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
        return self.final_output(f)

    # TODO: try deleting compile so that it's not overriden from original model base class method
    def compile(self, **kwargs) -> None:
        """Compile fuzzy network."""
        super().compile(**kwargs)

    def fit(self, *args, **kwargs):
        """Fit fuzzy network to training data."""
        if not self.built:
            logger.debug('FuzzyNetwork cannot be built until seeing training data')

        kwargs['sample_data'] = kwargs.get('sample_data', args[0])
        fuzzy_initializer_kwargs = parse_function_kwargs(kwargs, FuzzyWeightsInitializer.__init__)

        # add callback to instantiate fuzzy weights unless already provided
        if 'callbacks' in kwargs:
            if any([isinstance(cb, FuzzyWeightsInitializer) for cb in kwargs['callbacks']]):
                logger.warning('User already provided Fuzzy Weight Initializer callback. '
                               'Will use existing Fuzzy Weight Initializer in kwargs')
            else:
                kwargs['callbacks'].append(FuzzyWeightsInitializer(**fuzzy_initializer_kwargs))
        else:
            kwargs['callbacks'] = [FuzzyWeightsInitializer(**fuzzy_initializer_kwargs)]

        super().fit(*args, **parse_function_kwargs(kwargs, Model.fit))
        if not self.trained:
            self.trained = True

    def summary(self, *args, **kwargs):
        """Show summary of fuzzy network."""
        x = Input(shape=self.input_shape, name="InputRow")
        model = Model(inputs=[x], outputs=self.call(x), name=self.name + ' Summary')
        return model.summary(*args, **kwargs)

    def get_config(self):
        """Generate model config."""
        base_config = super().get_config()
        base_config['input_shape'] = self.input_shape
        base_config['neurons'] = self.neurons
        base_config['num_classes'] = self.num_classes
        activation = self.final_output.activation_function if isinstance(self.final_output.activation_function, str) \
            else serialize(self.final_output.activation_function)
        base_config.update({'activation': activation if isinstance(activation, str) else serialize(activation)})
        return base_config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        activation = config.pop('activation')
        if isinstance(activation, str):
            # get automatically deserializes to the function
            activation = get_activation(activation)
        return cls(activation=activation, **config)

    # TODO: add 'get_comple_config()' and 'compile_from_config(config)' to get rid of warning below:
    #       UserWarning: `compile()` was not called as part of model loading because the model's `compile()` method is custom.
    #       All subclassed Models that have `compile()` overridden should also override
    #       `get_compile_config()` and `compile_from_config(config)`.
    #       Alternatively, you can call `compile()` manually after loading.
