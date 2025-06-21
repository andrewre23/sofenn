import logging

import keras.api.ops as K
import numpy
from keras.api.callbacks import Callback
from numpy.typing import ArrayLike

from sofenn.layers import FuzzyLayer

logger = logging.getLogger(__name__)


class FuzzyWeightsInitializer(Callback):
    def __init__(
            self,
            sample_data,
            random_sample=True,
            s_0 = 4.0,
            layer_name: str = 'FuzzyRules'
    ):
        super().__init__()
        self.sample_data = sample_data
        self.random_sample = random_sample
        self.s_0 = s_0
        self.layer_name = layer_name

    def on_train_begin(self, logs=None):
        logger.debug('Initializing Fuzzy Weights prior to training.')

        if not isinstance(self.model.get_layer(self.layer_name), FuzzyLayer):
            raise ValueError(f'Initializer must be used on FuzzyLayer. Attempted to use: '
                             f'name={self.layer_name} type={type(self.model.get_layer(self.layer_name))}.')

        # build fuzzy rules layer before initializing centers
        if not self.model.get_layer(self.layer_name).built:
            self.model.get_layer(self.layer_name).build(input_shape=self.sample_data.shape)

        # initialize centers for first training run
        if not self.model.trained:
            self._initialize_centers(sample_data=self.sample_data, random_sample=self.random_sample)
            self._initialize_widths(s_0=self.s_0)

    def _initialize_centers(self, sample_data: ArrayLike, random_sample: bool = True) -> None:
        """
        Initialize neuron center weights with samples from sample dataset.

        :param sample_data: Array of sample data.
        :param random_sample: If True, randomly sample weights. If False, take first n (# neurons) records.
        """
        if random_sample:
            # set centers as random sampled index values
            samples = numpy.random.randint(0, len(sample_data), self.model.neurons)
            x_i = numpy.array([sample_data[samp] for samp in samples])
        else:
            # take first few samples, one for each neuron
            x_i = sample_data[:self.model.neurons]

        # reshape from (neurons, features) to (features, neurons)
        c_init = x_i.T

        # set weights
        c, s = self.model.get_layer(self.layer_name).get_weights()
        start_weights = [c_init, s]
        self.model.get_layer(self.layer_name).set_weights(start_weights)

    def _initialize_widths(self, s_0: float = 4.0)  -> None:
        """
        Initialize neuron widths.

        :param s_0: Initial sigma value for neuron centers.
        """
        # get current center and width weights
        c, s = self.model.get_layer(self.layer_name).get_weights()

        # repeat s_0 value to array shaped like s
        s_init = K.repeat(s_0, s.size).numpy().reshape(s.shape)

        # set weights
        start_weights = [c, s_init]
        self.model.get_layer(self.layer_name).set_weights(start_weights)
