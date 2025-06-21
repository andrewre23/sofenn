from typing import Optional

import keras.api.ops as K
import keras.src.backend as k
from keras.api.layers import Layer

from sofenn.utils.layers import get_fuzzy_output_shape


class FuzzyLayer(Layer):
    """
    Fuzzy Layer (1) of SOFNN
    ========================

    - Radial (Ellipsoidal) Basis Function Layer

    - each neuron represents "if-part" or premise of a fuzzy rule
    - individual Membership Functions (MF) are applied to each feature for each neuron
    - output is product of Membership Functions
    - each MF is Gaussian function:

        - mu(i,j) = exp{- [x(i) - c(i,j)]^2 / [2 * sigma(i,j)^2]}

        - for i features and  j neurons:

        - mu(i,j)    = ith MF of jth neuron

        - c(i,j)     = center of ith MF of jth neuron

        - sigma(i,j) = width of ith MF of jth neuron

    - output for Fuzzy Layer is:
        phi(j) = exp{-sum[i=1,r;
                    [x(i) - c(i,j)]^2 / [2 * sigma(i,j)^2]]}
    """
    def __init__(self,
                 shape: tuple,
                 neurons: Optional[int] = 1,
                 initializer_centers: Optional[str] = 'uniform',
                 initializer_sigmas: Optional[str] = 'ones',
                 name: Optional[str] = "FuzzyRules",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if neurons <= 0:
            raise ValueError("Neurons must be a positive integer.")
        self.neurons = neurons
        self.features = shape[-1]
        self.shape = get_fuzzy_output_shape(shape, neurons)
        self.initializer_centers = initializer_centers
        self.initializer_sigmas = initializer_sigmas
        self.c = None
        self.s = None
        self.built = False

    def build(self, input_shape: tuple, **kwargs) -> None:
        """
        Build objects for processing steps.

        Parameters
        ==========
        input_shape : tuple
            - input shape of training data
            - last index will be taken for sizing variables

        Attributes
        ==========
        c : center
            - c(i,j)
            - trainable weights for center of ith membership function of jth neuron
            - shape: (features, neurons)

        s : sigma
            - s(i,j)
            - trainable weights for width of ith membership function of jth neuron
            - shape: (features, neurons)
        """
        self.c = self.add_weight(name='c',
                                 shape=(self.features, self.neurons),
                                 initializer=self.initializer_centers,
                                 trainable=True,
                                 **kwargs)
        self.s = self.add_weight(name='s',
                                 shape=(self.features, self.neurons),
                                 initializer=self.initializer_sigmas,
                                 trainable=True,
                                 **kwargs)
        super().build(input_shape, **kwargs)

    def call(self, inputs: k.KerasTensor, **kwargs) -> k.KerasTensor:
        """
        Build processing logic for layer.

        Parameters
        ==========
        inputs : tensor
            - input tensor
            - shape: (samples,features)

        Attributes
        ==========
        aligned_x : tensor
            - x(i,j)
            - ith feature of jth neuron
            - shape: (samples, features, neurons)

        aligned_c : tensor
            - c(i,j)
            - center of ith membership function of jth neuron
            - shape: (features, neurons)

        aligned_s : tensor
            - s(i,j)
            - sigma of ith membership function of jth neuron
            - shape: (features, neurons)

        Returns
        =======
        phi: tensor
            - phi(neurons,)
            - output of jth neuron in fuzzy layer
            - shape: (samples, neurons)
        """
        if not self.built:
            self.build(input_shape=inputs.shape, **kwargs)

        aligned_x = K.repeat(K.expand_dims(inputs, axis=-1), self.neurons, -1)
        aligned_c = self.c
        aligned_s = self.s

        # calculate output of each neuron (fuzzy rule)
        x_minus_c_squared = K.square(aligned_x - aligned_c)
        two_sigma = 2 * K.square(aligned_s)
        phi = K.exp(-K.sum(K.true_divide(x_minus_c_squared, two_sigma),
                           axis=-2, keepdims=False))
        return phi

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return output shape of input data.

        Parameters
        ==========
        input_shape : tuple
            - shape of input data
            - shape: (samples, features)

        Returns
        =======
        output_shape : tuple
            - output shape of fuzzy layer
            - shape: (samples, neurons)
        """
        return tuple(input_shape[:-1]) + (self.neurons,)

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer.
        """
        base_config = super(FuzzyLayer, self).get_config()
        base_config['neurons'] = self.neurons
        base_config['shape'] = self.shape
        base_config['initializer_centers'] = self.initializer_centers
        base_config['initializer_sigmas'] = self.initializer_sigmas
        return base_config
