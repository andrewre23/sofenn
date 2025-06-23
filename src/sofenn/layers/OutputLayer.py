from typing import Optional

import keras.api.ops as K
import keras.src.backend as k
from keras.api.activations import softmax, linear, sigmoid
from keras.api.layers import Layer, Dense


class OutputLayer(Layer):
    """
    Output Layer
    ============
    Final output for Fuzzy Neural Network

    Layer (4) of SOFNN Model

    Unweighted sum of each output from previous layer (f)

    with:
      - *i* = features
      - *j* = neurons
      - .. math::
        i=1,2,...,r;
      - .. math::
        j=1,2,...,u;


    Output is:
    .. math::
        \sum{k=1}^{u} f_{(k}}



    - shape: (samples,)
    """

    def __init__(
            self,
            shape: tuple,
            target_classes: int,
            problem_type: str,
            name: Optional[str] = "Outputs",
            **kwargs
    ):
        defaults = {
                'classification': softmax,
                'regression': linear,
                'logistic_regression': sigmoid,
            }

        super().__init__(name=name, **kwargs)
        self.shape = k.standardize_shape(shape)
        self.target_classes = target_classes
        self.problem_type = problem_type
        self.activation = Dense(
            units=self.target_classes,
            name=defaults[problem_type].__name__.capitalize(),
            activation=defaults[problem_type],
            dtype='float32'
        )
        self.output_dim = target_classes
        self.built = True

    def build(self, input_shape: tuple, **kwargs) -> None:
        """
        Build objects for processing steps.

        Parameters
        ==========
        input_shape: tuple
            - f shape: (samples, neurons)
        """
        super().build(input_shape=input_shape, **kwargs)

    def call(self, inputs: k.KerasTensor) -> k.KerasTensor:
        """
        Build processing logic for layer.

        Parameters
        ==========
        inputs: tensor
            - tensor with f as output of previous layer
            - f shape: (samples, neurons)

        Returns
        =======
        output: tensor
            - sum of all f's from previous layer
            - shape: (samples,)
        """
        # get raw sum of all neurons for each sample
        sums = K.sum(inputs, axis=-1)
        output = K.expand_dims(sums, axis=-1)

        final_output = self.activation(K.expand_dims(output, axis=-1) if len(output.shape) == 1 else output)
        if self.problem_type == 'regression':
            return K.squeeze(final_output, axis=-1)
        # elif self.problem_type == 'classification':
        #     return K.squeeze(final_output, axis=0) if final_output.shape[0] == 1 else final_output
        else:
            return final_output

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return output shape of input data.

        Parameters
        ==========
        input_shape: tuple
            - f shape: (samples, neurons)

        Returns
        =======
        output_shape: tuple
            - output shape of final layer
            - shape: (samples,)
        """
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> dict:
        """
        Return config dictionary for custom layer.
        """
        base_config = super(OutputLayer, self).get_config()
        base_config['shape'] = self.shape
        base_config['target_classes'] = self.target_classes
        base_config['problem_type'] = self.problem_type
        return base_config
