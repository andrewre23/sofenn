import numpy as np
from keras.api.callbacks import Callback

from sofenn.layers import FuzzyLayer


class FuzzyWeightsInitializer(Callback):
    def __init__(self,
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
        # TODO: update to logging
        #print("Initializing Fuzzy Weights prior to training...")
        # print(f"params attribute: {self.params}")
        # print(f'Model Status: {self.model.built}')
        # print(f'Fuzzy rules layer Status: {self.model.get_layer("FuzzyRules")}')
        # print(f'Fuzzy rules weights: {self.model.get_layer("FuzzyRules").get_weights()}')
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

    def on_train_end(self, logs=None):
        pass
        # TODO: update to logging
        # print("...post training")
        # print(f'Model Status: {self.model.built}')
        # print(f'Fuzzy rules layer Status: {self.model.get_layer("FuzzyRules")}')
        # print(f'Fuzzy rules weights: {self.model.get_layer("FuzzyRules").get_weights()}')

    def _initialize_centers(self,
                            sample_data: np.ndarray,
                            random_sample: bool = True
                            ) -> None:
        """
        Initialize neuron center weights with samples from X_train dataset.

        Parameters
        ==========
        random: bool
            - take random samples from training data or
            take first n instances (n=# of neurons)
        """
        if random_sample:
            # set centers as random sampled index values
            samples = np.random.randint(0, len(sample_data), self.model.neurons)
            x_i = np.array([sample_data[samp] for samp in samples])
        else:
            # take first few samples, one for each neuron
            x_i = sample_data[:self.model.neurons]

        # reshape from (neurons, features) to (features, neurons)
        c_init = x_i.T

        # set weights
        c, s = self.model.get_layer(self.layer_name).get_weights()
        start_weights = [c_init, s]
        self.model.get_layer(self.layer_name).set_weights(start_weights)
        # validate weights updated as expected
        final_weights = self.model.get_layer(self.layer_name).get_weights()
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])

    def _initialize_widths(self, s_0: float = 4.0)  -> None:
        """
        Initialize neuron widths.

        Parameters
        ==========
        s_0 : float
            - initial sigma value for all neuron centers
        """
        # get current center and width weights
        c, s = self.model.get_layer(self.layer_name).get_weights()

        # repeat s_0 value to array shaped like s
        s_init = np.repeat(s_0, s.size).reshape(s.shape)

        # set weights
        start_weights = [c, s_init]
        self.model.get_layer(self.layer_name).set_weights(start_weights)
        # validate weights updated as expected
        final_weights = self.model.get_layer(self.layer_name).get_weights()
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])
