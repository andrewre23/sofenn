from keras import backend as K
from keras.engine.topology import Layer


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

    def __init__(self,
                 **kwargs):
        # adjust argumnets
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        # default Name
        if 'name' not in kwargs:
            kwargs['name'] = 'RawOutput'
        self.output_dim = 1
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Build objects for processing steps

        Parameters
        ==========
        input_shape : tuple
            - f shape : (samples, neurons)
        """
        super().build(input_shape)

    def call(self, x, **kwargs):
        """
        Build processing logic for layer

        Parameters
        ==========
        x : tensor
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
        sums = K.sum(x, axis=-1)
        return K.repeat_elements(K.expand_dims(sums, axis=-1), self.output_dim, -1)

    def compute_output_shape(self, input_shape):
        """
        Return output shape of input data

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
