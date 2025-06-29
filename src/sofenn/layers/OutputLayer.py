from typing import Optional, Union, Callable

import keras
import keras.ops as k
from keras import activations
from keras.layers import Layer, Dense

from sofenn.utils.layers import replace_last_dim, is_valid_activation


@keras.saving.register_keras_serializable()
class OutputLayer(Layer):
    r"""
    Output Layer
    ============
    Final output for Fuzzy Neural Network

    Layer (4) of SOFNN Model

    Unweighted sum of each output from previous layer (f)

    with:
      - *i* = features
      - *j* = neurons
      - .. math::
        i=1,2,...,r;
      - .. math::
        j=1,2,...,u;


    Output is:
    .. math::
        \sum{k=1}^{u} f_{(k}}

    - shape: (*,)

    :param num_classes: Number of classes in output data.
    :param activation: Activation function to use (default: linear)..
    :param name: Layer name (default: Outputs).
    """
    def __init__(
            self,
            num_classes: int = 1,
            activation: Union[str, Callable] = activations.linear,
            name: Optional[str] = 'Outputs',
            **kwargs
    ):
        if not is_valid_activation(activation):
            raise ValueError(f"Invalid activation function: '{activation}'")
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.activation_function = activation
        self.activation_layer = Dense(
            units=num_classes,
            name=activation.__name__.capitalize() if callable(activation) else activation,
            activation=activation,
            use_bias=activation not in ['linear', activations.linear], # simplify the model by dropping bias when linear
            dtype='float32'
        )
        self.built = False

    def call(self, inputs: keras.KerasTensor, **kwargs) -> keras.KerasTensor:
        """
        Build processing logic for layer.

        Parameters
        ==========
        inputs: tensor
            - tensor with f as output of previous layer
            - f shape: (*, neurons)

        Returns
        =======
        output: tensor
            - sum of all f's from previous layer
            - shape: (*, num_classes)
        """
        # get the raw sum of all neurons for each sample
        sums = k.sum(inputs, axis=-1, keepdims=True)
        # ndim > 2 required for passing through activation
        output = k.expand_dims(sums, 0) if sums.ndim < 2 else sums
        return self.activation_layer(output)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return output shape of input data.

        Parameters
        ==========
        input_shape: tuple
            - f shape: (*, neurons)

        Returns
        =======
        output_shape: tuple
            - output shape of final layer
            - shape: (*, num_classes)
        """
        return replace_last_dim(input_shape, self.num_classes)

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer.
        """
        base_config = super(OutputLayer, self).get_config()
        base_config['num_classes'] = self.num_classes
        base_config.update({
            'activation': self.activation_function if isinstance(self.activation_function, str) \
                else activations.serialize(self.activation_function)
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        activation = config.pop('activation')
        if isinstance(activation, str):
            # get automatically deserializes to the function
            activation = activations.get(activation)
        return cls(activation=activation, **config)
