from typing import Optional

import keras
import keras.ops as k
from keras.layers import Layer


@keras.saving.register_keras_serializable()
class NormalizeLayer(Layer):
    r"""
    Normalize Layer
    ===============
    Normalization Layer

    Layer (2) of SOFNN Model

    Output of each neuron is normalized by total output from the previous layer

    Number of outputs equal to the previous layer (# of neurons)

    with:
        - *j* = neurons
        - .. math::
            j=1,2,...,u;

    Output for Normalize Layer is:
        .. math::
            \psi_{(j)} = \phi_{(j)} / \sum_{k=1}^{u} \phi_{(k)}

    :param name: Layer name (default: Normalize).
    """
    def __init__(self, name: Optional[str] = "Normalize", **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_shape = None
        self.fixed_weight = None
        self.built = False

    def call(self, inputs: keras.KerasTensor, **kwargs) -> keras.KerasTensor:
        """
        Build processing logic for layer

        Parameters
        ==========
        inputs: tensor
            - input tensor
            - tensor with phi output of each neuron
            - phi(j) for j neurons
            - shape: (*, neurons)

        Returns
        =======
        psi: tensor
            - output of each neuron after normalization step
            - divide each output by sum of output of all neurons
            - psi(j) for jth neuron
            - shape: (*, neurons)
        """
        sums = k.sum(inputs, axis=-1)
        sums_expanded = k.repeat(k.expand_dims(sums, axis=-1), inputs.shape[-1], -1)
        return inputs / sums_expanded

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return output shape of input data.

        Parameters
        ==========
        input_shape: tuple
            - shape of input data
            - shape: (*, neurons)

        Returns
        =======
        output_shape: tuple
            - output shape of normalization layer
            - shape: (*, neurons)
        """
        return input_shape

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer
        """
        return super(NormalizeLayer, self).get_config()
