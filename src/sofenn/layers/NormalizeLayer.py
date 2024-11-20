import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer


class NormalizedLayer(Layer):
    """
    Normalized Layer (2) of SOFNN
    =============================

    - Normalization Layer

    - output of each neuron is normalized by total output from previous layer
    - number of outputs equal to previous layer (# of neurons)
    - output for Normalized Layer is:

        psi(j) = phi(j) / sum[k=1, u; phi(k)]
                for u neurons
        - with:

        psi(j) = output of Fuzzy Layer neuron j
    """

    def __init__(self,
                 output_dim: int,
                 **kwargs):
        # adjust argumnets
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        # default Name
        if 'name' not in kwargs:
            kwargs['name'] = 'Normalization'
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape: tuple) -> None:
        """
        Build objects for processing steps

        Parameters
        ==========
        input_shape : tuple
            - input shape of training data
            - last index will be taken for sizing variables

        """
        super().build(input_shape)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Build processing logic for layer

        Parameters
        ==========
        x : tensor
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
        sums = K.sum(x, axis=-1)
        sums = K.repeat_elements(K.expand_dims(sums, axis=-1), self.output_dim, -1)

        # assert tensor shapes
        assert(x.shape[-1] == sums.shape[-1])

        return x / sums

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return output shape of input data

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
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer

        """
        base_config = super(NormalizedLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config
