#
# SOFENN
# Self-Organizing Fuzzy Neural Network
#
# (sounds like soften)
#
#
# Implemented per description in
# An on-line algorithm for creating self-organizing
# fuzzy neural networks
# Leng, Prasad, McGinnity (2004)
#
#
# Andrew Edmonds - 2019
# github.com/andrewre23
#


from keras import backend as K
from keras.engine.topology import Layer


class DefuzzyLayer(Layer):

    def __init__(self,
                 output_dim,
                 initializer_rules_outcome=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.initializer_rules_outcome = initializer_rules_outcome
        super(DefuzzyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dimensions = list(input_shape)[:-1:-1]
        self.rules_outcome = self.add_weight(name='rules_outcome',
                                             shape=(input_shape[1], self.output_dim),
                                             initializer=self.initializer_rules_outcome if
                                             self.initializer_rules_outcome is not None else 'uniform',
                                             trainable=True)

        super(DefuzzyLayer, self).build(input_shape)

    def call(self, x):
        aligned_x = K.repeat_elements(K.expand_dims(x, axis=-1), self.output_dim, -1)
        aligned_rules_outcome = self.rules_outcome
        for dim in self.input_dimensions:
            aligned_rules_outcome = K.repeat_elements(K.expand_dims(aligned_rules_outcome, 0), dim, 0)

        xc = K.sum((aligned_x * aligned_rules_outcome), axis=-2, keepdims=False)
        return xc

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)
