from typing import List, Optional

import keras
import keras.ops as K
from keras.layers import Layer


@keras.saving.register_keras_serializable()
class WeightedLayer(Layer):
    r"""
    Weighted Layer
    ==============
    Weighted Layer

    Layer (3) of SOFNN Model

    Weighting of ith MF of each feature

    Yields the "consequence" of the jth fuzzy rule of fuzzy model

    Each neuron has two inputs:
        - output of previous related neuron j
        - weighted bias w2j

    with:
      - *r* = number of original input features
    .. math::
      - B = [1, x_{1}, x_{2}, ... x_{r}]

      - A_{j} = [a_{j0}, a_{j1}, a_{j2}, ... a_{r}]

      - w_{2j} = Aj * B = a_{j0} + a_{j1}x_{1} + a_{j2}x_{2} + ... + a_{jr}x_{r}


    Output for weighted layer is:
        .. math::
            f_{j} = w_{2j} \psi_{(j)}

    :param initializer_a: Initializer for A matrix of weighted layer
    :param name: Name for keras Model.
    """
    def __init__(self,
                 initializer_a: Optional[str] = 'uniform',
                 name: Optional[str] = 'Weights',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.initializer_a = initializer_a
        self.a = None
        self.built = False

    def build(self, input_shape: List[tuple], **kwargs) -> None:
        """
        Build objects for processing steps.

        Parameters
        ==========
        input_shape: list of tuples
            - [x shape, psi shape]
            - x shape: (*, features)
            - psi shape: (*, neurons)

        Attributes
        ==========
        a: then-part (consequence) of fuzzy rule
            - a(j, i)
            - trainable weight of ith feature of jth neuron
            - shape: (neurons, 1+features)
        """
        x_shape, psi_shape = input_shape
        input_dim = 1 if x_shape[-1] is None else x_shape[-1]
        output_dim = 1 if psi_shape[-1] is None else psi_shape[-1]

        self.a = self.add_weight(name='a',
                                 shape=(output_dim, 1+input_dim),
                                 initializer=self.initializer_a if
                                 self.initializer_a is not None else 'uniform',
                                 trainable=True,
                                 **kwargs)
        super().build(input_shape, **kwargs)

    def call(self, inputs: List[keras.KerasTensor], **kwargs) -> keras.KerasTensor:
        """
        Build processing logic for layer.

        Parameters
        ==========
        inputs: list of tensors
            - list of tensor with input data and psi output of previous layer
            - [x, psi]
            - x shape: (*, features)
            - psi shape: (*, neurons)

        Attributes
        ==========
        aligned_b: tensor
            - input vector with [1.0] prepended for bias weight
            - shape: (*, 1+features)

        aligned_a: tensor
            - a(i,j)
            - weight parameter of ith feature of jth neuron
            - shape: (neurons, 1+features)

        Returns
        =======
        f: tensor
            - psi(neurons,)
            - output of each neuron in fuzzy layer
            - shape: (*, neurons)
        """
        # TODO: remove forcing build before first call. should be happening automatically
        if not self.built:
            self.build(input_shape=[x_or_psi.shape for x_or_psi in inputs], **kwargs)

        x, psi = inputs

        # align tensors by prepending bias value for input tensor in b
        # b shape: (*, 1+features)
        b = K.mean(K.ones_like(x), -1, keepdims=True)
        aligned_b = K.concatenate([b, x], axis=-1)
        aligned_a = self.a
        w2 = K.dot(aligned_a, K.transpose(aligned_b))
        return K.multiply(psi, K.transpose(w2))

    def compute_output_shape(self, input_shape: List[tuple]) -> tuple:
        """
        Return output shape of input data.

        Parameters
        ==========
        input_shape: list of tuples
            - [x, psi]
            - x shape: (*, features)
            - psi shape: (*, neurons)

        Returns
        =======
        output_shape: tuple
            - output shape of weighted layer
            - shape: (*, neurons)
        """
        x_shape, psi_shape = input_shape
        return tuple(x_shape[:-1]) + (psi_shape[-1],)

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer.
        """
        base_config = super(WeightedLayer, self).get_config()
        base_config['initializer_a'] = self.initializer_a
        return base_config
