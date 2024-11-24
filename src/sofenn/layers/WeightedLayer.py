from typing import Callable, List, Optional

import tensorflow as tf
from keras import backend as K
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
                normalized layer

    -output for weighted layer is:
        fj     = w2j psi(j)
    """

    def __init__(self,
                 output_dim: int,
                 initializer_a: Optional[Callable]=None,
                 **kwargs):
        # adjust argumnets
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        # default Name
        if 'name' not in kwargs:
            kwargs['name'] = 'Weights'
        self.output_dim = output_dim
        self.initializer_a = initializer_a
        super().__init__(**kwargs)

    def build(self, input_shape: List[tuple]) -> None:
        """
        Build objects for processing steps

        Parameters
        ==========
        input_shape : list of tuples
            - [x shape, psi shape]
            - x shape: (samples, features)
            - psi shape: (samples, neurons)

        Attributes
        ==========
        a : then-part (consequence) of fuzzy rule
            - a(i,j)
            - trainable weight of ith feature of jth neuron
            - shape: (1+features, neurons)
        """
        # assert multi-input as list
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2

        # extract variables
        x_shape, psi_shape = input_shape

        self.a = self.add_weight(name='a',
                                 shape=(1+x_shape[-1], self.output_dim),
                                 initializer=self.initializer_a if
                                 self.initializer_a is not None else 'uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Build processing logic for layer

        Parameters
        ==========
        x : list of tensors
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
            - shape: (1+features, neurons)

        Returns
        =======
        f: tensor
            - psi(neurons,)
            - output of each neuron in fuzzy layer
            - shape: (neurons,)
        """
        # assert multi-input as list and read in inputs
        assert isinstance(x, list)
        assert len(x) == 2
        x, psi = x

        # align tensors by prepending bias value for input tensor in b
        # b shape: (samples, 1)
        b = K.ones((K.tf.shape(x)[0], 1), dtype=x.dtype)
        aligned_b = K.concatenate([b, x])
        aligned_a = self.a

        # assert input and weight vectors are compatible
        # w2 shape: (samples, neurons)
        assert(aligned_b.shape[-1] == aligned_a.shape[0])
        w2 = K.tf.matmul(aligned_b, aligned_a)

        # assert psi and resulting w2 vector are compatible
        assert (psi.shape[-1] == w2.shape[-1])

        return psi * w2

    def compute_output_shape(self, input_shape: List[tuple]) -> tuple:
        """
        Return output shape of input data

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
        # assert multi-input as list
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2
        x_shape, psi_shape = input_shape

        return tuple(x_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer

        """
        base_config = super(WeightedLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config
