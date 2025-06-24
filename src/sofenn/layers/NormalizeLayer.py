from typing import Optional

import keras
import keras.ops as K
import keras.src.backend as k
from keras.layers import Layer


@keras.saving.register_keras_serializable()
class NormalizeLayer(Layer):
    """
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
        input_shape: tuple
            - input shape of training data
            - last index will be taken for sizing variables
        """
        super().build(input_shape=input_shape, **kwargs)

    def call(self, inputs: k.KerasTensor) -> k.KerasTensor:
        """
        Build processing logic for layer

        Parameters
        ==========
        inputs: tensor
            - input tensor
            - tensor with phi output of each neuron
            - phi(j) for j neurons
            - shape: (samples, neurons)

        Returns
        =======
        psi: tensor
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
        input_shape: tuple
            - shape of input data
            - shape: (samples, neurons)

        Returns
        =======
        output_shape: tuple
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
