from typing import Optional

import keras
import keras.ops as k
from keras.layers import Layer

from sofenn.utils.layers import replace_last_dim


@keras.saving.register_keras_serializable()
class FuzzyLayer(Layer):
    r"""
    Fuzzy Layer
    ===========
    Radial (Ellipsoidal) Basis Function Layer

    Layer (1) of SOFNN Model

    Each neuron represents "if-part" or premise of a fuzzy rule

    Individual Membership Functions are applied to each feature for each neuron

    Output is product of Membership Functions

    Each **Membership Function (MF)** is Gaussian function:

    .. math::
        - \mu_{(i,j)} = exp^{- \dfrac{{[x_{(i)} - c_{(i,j)}]}^2}{[2 * \sigma_{(i,j)}^{2}]}}

    with:
      - *i* = features
      - *j* = neurons
      - .. math::
        i=1,2,...,r;
      - .. math::
        j=1,2,...,u;
      - center of ith MF of jth neuron
            .. math::
                c_{(i,j)}
      - width of ith MF of jth neuron
            .. math::
                \sigma_{(i,j)}

    Output for Fuzzy Layer is:
        .. math::
            \phi_{(j)} = \sum_{j=1}^{u} \dfrac{{[x_{(i)} - c_{(i,j)}]}^2}{[2 * \sigma_{(i,j)}^{2}]}

    :param neurons: Number of fuzzy neurons.
    :param initializer_centers: Initializer for fuzzy centers.
    :param initializer_sigmas: Initializer for fuzzy widths.
    :param name: Name for keras Model.
    """
    def __init__(self,
                 neurons: Optional[int] = 1,
                 initializer_centers: Optional[str] = 'uniform',
                 # TODO: determine if better to refer to these as 'widths' or 'sigmas'
                 initializer_sigmas: Optional[str] = 'ones',
                 name: Optional[str] = "FuzzyRules",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        if neurons <= 0:
            raise ValueError("Neurons must be a positive integer.")
        self.input_shape = None
        self.neurons = neurons
        self.initializer_centers = initializer_centers
        self.initializer_sigmas = initializer_sigmas
        self.c = None
        self.s = None
        self.built = False

    def build(self, input_shape: tuple, **kwargs) -> None:
        r"""
        Build objects for processing steps.

        Parameters
        ==========
        input_shape: tuple
            - input shape of training data
            - last index will be taken for sizing variables

        Attributes
        ==========
        c: center
            - .. math:: c_{(i,j)}
            - trainable weights for center of ith membership function of jth neuron
            - shape: (features, neurons)

        s: width / sigma
            - .. math:: \sigma_{(i,j)}
            - trainable weights for width of ith membership function of jth neuron
            - shape: (features, neurons)
        """
        self.input_shape = input_shape
        features = input_shape[-1]

        self.c = self.add_weight(name='c',
                                 shape=(features, self.neurons),
                                 initializer=self.initializer_centers,
                                 trainable=True,
                                 **kwargs)
        self.s = self.add_weight(name='s',
                                 shape=(features, self.neurons),
                                 initializer=self.initializer_sigmas,
                                 trainable=True,
                                 **kwargs)
        super().build(input_shape, **kwargs)

    def call(self, inputs: keras.KerasTensor, **kwargs) -> keras.KerasTensor:
        """
        Build processing logic for layer.

        Parameters
        ==========
        inputs: tensor
            - input tensor
            - shape: (*, features)

        Attributes
        ==========
        aligned_x: tensor
            - x(i,j)
            - ith feature of jth neuron
            - shape: (*, features, neurons)

        aligned_c: tensor
            - c(i,j)
            - center of ith membership function of jth neuron
            - shape: (features, neurons)

        aligned_s: tensor
            - s(i,j)
            - sigma of ith membership function of jth neuron
            - shape: (features, neurons)

        Returns
        =======
        phi: tensor
            - phi(neurons,)
            - output of jth neuron in fuzzy layer
            - shape: (*, neurons)
        """
        # TODO: remove forcing build before first call. should be happening automatically
        if not self.built:
            self.build(input_shape=inputs.shape, **kwargs)

        aligned_x = k.repeat(k.expand_dims(inputs, axis=-1), self.neurons, -1)
        aligned_c = self.c
        aligned_s = self.s

        # calculate output of each neuron (fuzzy rule)
        x_minus_c_squared = k.square(aligned_x - aligned_c)
        two_sigma = 2 * k.square(aligned_s)
        phi = k.exp(-k.sum(k.true_divide(x_minus_c_squared, two_sigma),
                           axis=-2, keepdims=False))
        return phi

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return output shape of input data.

        Parameters
        ==========
        input_shape: tuple
            - shape of input data
            - shape: (*, features)

        Returns
        =======
        output_shape: tuple
            - output shape of fuzzy layer
            - shape: (*, neurons)
        """
        return replace_last_dim(input_shape, self.neurons)

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer.
        """
        base_config = super(FuzzyLayer, self).get_config()
        base_config['neurons'] = self.neurons
        base_config['initializer_centers'] = self.initializer_centers
        base_config['initializer_sigmas'] = self.initializer_sigmas
        return base_config
