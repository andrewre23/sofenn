from typing import List, Optional

import keras.api.ops as K
import keras.src.backend as k
from keras.api.layers import Layer


class WeightedLayer(Layer):
    """
    Weighted Layer (3) of SOFNN
    ===========================

    - Weighting of ith MF of each feature

    - yields the "consequence" of the jth fuzzy rule of fuzzy model
    - each neuron has two inputs:
        - output of previous related neuron j
        - weighted bias w2j
    - with:
        r      = number of original input features

        B      = [1, x1, x2, ... xr]
        Aj     = [aj0, aj1, ... ajr]

        w2j    = Aj * B =
                 aj0 + aj1x1 + aj2x2 + ... ajrxr

        psi(j) = output of jth neuron from
                normalize layer

    -output for weighted layer is:
        fj     = w2j psi(j)
    """
    def __init__(self,
                 shape: List[tuple],
                 initializer_a: Optional[str] = 'uniform',
                 name: Optional[str] = 'Weights',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        x_shape, psi_shape = shape
        self.x_shape = k.standardize_shape(x_shape)
        self.psi_shape = k.standardize_shape(psi_shape)
        self.output_dim = psi_shape[-1]
        self.initializer_a = initializer_a
        self.a = None
        self.built = False

    def build(self, input_shape: List[tuple], **kwargs) -> None:
        """
        Build objects for processing steps.

        Parameters
        ==========
        input_shape : list of tuples
            - [x shape, psi shape]
            - x shape: (samples, features)
            - psi shape: (samples, neurons)

        Attributes
        ==========
        a : then-part (consequence) of fuzzy rule
            - a(j, i)
            - trainable weight of ith feature of jth neuron
            - shape: (neurons, 1+features)
        """
        x_shape, psi_shape = input_shape

        self.a = self.add_weight(name='a',
                                 shape=(self.output_dim, 1+x_shape[-1]),
                                 initializer=self.initializer_a if
                                 self.initializer_a is not None else 'uniform',
                                 trainable=True,
                                 **kwargs)
        super().build(input_shape, **kwargs)

    def call(self, inputs: List[k.KerasTensor], **kwargs) -> k.KerasTensor:
        """
        Build processing logic for layer.

        Parameters
        ==========
        inputs : list of tensors
            - list of tensor with input data and psi output of previous layer
            - [x, psi]
            - x shape: (samples, features)
            - psi shape: (samples, neurons)

        Attributes
        ==========
        aligned_b : tensor
            - input vector with [1.0] prepended for bias weight
            - shape: (samples, 1+features)

        aligned_a : tensor
            - a(i,j)
            - weight parameter of ith feature of jth neuron
            - shape: (neurons, 1+features)

        Returns
        =======
        f: tensor
            - psi(neurons,)
            - output of each neuron in fuzzy layer
            - shape: (samples, neurons)
        """
        if not self.built:
            self.build(input_shape=[x_or_psi.shape for x_or_psi in inputs], **kwargs)

        x, psi = inputs

        # align tensors by prepending bias value for input tensor in b
        # b shape: (samples, 1+features)
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
        input_shape : list of tuples
            - [x, psi]
            - x shape: (samples, features)
            - psi shape: (samples, neurons)

        Returns
        =======
        output_shape : tuple
            - output shape of weighted layer
            - shape: (samples, neurons)
        """
        x_shape, psi_shape = input_shape

        return tuple(x_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer.
        """
        base_config = super(WeightedLayer, self).get_config()
        base_config['shape'] = [self.x_shape, self.psi_shape]
        base_config['initializer_a'] = self.initializer_a
        return base_config
