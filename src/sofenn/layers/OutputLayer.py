from typing import Optional

import keras.api.ops as K
import keras.src.backend as k
from keras.api.layers import Layer


class OutputLayer(Layer):
    """
    Output Layer (4) of SOFNN
    ==========================

    - Unweighted sum of each output of previous layer (f)

    - output for fuzzy layer is:
        sum[k=1, u; f(k)]
                for u neurons
        - shape: (samples,)
    """

    def __init__(self, name: Optional[str] = "Outputs", **kwargs):        # adjust arguments
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(name=name, **kwargs)
        self.output_dim = 1
        self.built = True

    def build(self, input_shape: tuple, **kwargs) -> None:
        """
        Build objects for processing steps.

        Parameters
        ==========
        input_shape : tuple
            - f shape : (samples, neurons)
        """
        super().build(input_shape=input_shape, **kwargs)

    def call(self, inputs: k.KerasTensor) -> k.KerasTensor:
        """
        Build processing logic for layer.

        Parameters
        ==========
        inputs : tensor
            - tensor with f as output of previous layer
            - f shape: (samples, neurons)

        Returns
        =======
        output: tensor
            sum[k=1, u; f(k)]
                for u neurons
        - sum of all f's from previous layer
            - shape: (samples,)
        """
        # get raw sum of all neurons for each sample
        sums = K.sum(inputs, axis=-1)
        return K.repeat(K.expand_dims(sums, axis=-1), self.output_dim, -1)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return output shape of input data.

        Parameters
        ==========
        input_shape : tuple
            - f shape: (samples, neurons)

        Returns
        =======
        output_shape : tuple
            - output shape of final layer
            - shape: (samples,)
        """
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer.
        """
        base_config = super(OutputLayer, self).get_config()
        return base_config
