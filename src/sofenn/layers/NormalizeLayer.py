from typing import Optional

import keras.api.ops as K
import keras.src.backend as k
from keras.api.layers import Layer


class NormalizeLayer(Layer):
    """
    Normalize Layer (2) of SOFNN
    =============================

    - Normalization Layer

    - output of each neuron is normalized by total output from previous layer
    - number of outputs equal to previous layer (# of neurons)
    - output for Normalize Layer is:

        psi(j) = phi(j) / sum[k=1, u; phi(k)]
                for u neurons
        - with:

        psi(j) = output of Fuzzy Layer neuron j
    """

    def __init__(self, shape: tuple, name: Optional[str] = "Normalize", **kwargs):
        super().__init__(name=name, **kwargs)
        shape = k.standardize_shape(shape)
        self.shape = shape
        self.output_dim = self.shape[-1]
        self.built = True

    def build(self, input_shape: tuple, **kwargs) -> None:
        """
        Build objects for processing steps

        Parameters
        ==========
        input_shape : tuple
            - input shape of training data
            - last index will be taken for sizing variables
        """
        super().build(input_shape=input_shape, **kwargs)

    def call(self, inputs: k.KerasTensor) -> k.KerasTensor:
        """
        Build processing logic for layer

        Parameters
        ==========
        inputs : tensor
            - input tensor
            - tensor with phi output of each neuron
            - phi(j) for j neurons
            - shape: (samples, neurons)

        Returns
        =======
        psi : tensor
            - output of each neuron after normalization step
            - divide each output by sum of output of all neurons
            - psi(j) for jth neuron
            - shape: (samples, neurons)
        """
        sums = K.sum(inputs, axis=-1)
        sums = K.repeat(K.expand_dims(sums, axis=-1), self.output_dim, -1)

        return inputs / sums

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return output shape of input data.

        Parameters
        ==========
        input_shape : tuple
            - shape of input data
            - shape: (samples, neurons)

        Returns
        =======
        output_shape : tuple
            - output shape of normalization layer
            - shape: (samples, neurons)
        """
        return input_shape

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer
        """
        base_config = super(NormalizeLayer, self).get_config()
        base_config['shape'] = self.shape
        return base_config
