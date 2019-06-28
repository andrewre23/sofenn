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

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical

from sklearn.metrics import mean_absolute_error

# custom Fuzzy Layers
from .layers import FuzzyLayer, NormalizedLayer, WeightedLayer, OutputLayer


class FuzzyNetwork(object):
    """
    Fuzzy Network
    =============

    -Implemented per description in:
        "An on-line algorithm for creating self-organizing
        fuzzy neural networks" - Leng, Prasad, McGinnity (2004)
    -Composed of 5 layers with varying "fuzzy rule" nodes

    * = samples

    Parameters
    ==========
    - X_train : training input data
        - shape :(train_*, features)
    - X_test  : testing input data
        - shape: (test_*, features)
    - y_train : training output data
        - shape: (train_*,)
    - y_test  : testing output data
        - shape: (test_*,)

    Attributes
    ==========
    - prob_type : str
        - regression/classification problem
    - classes : int
        - number of output classes for classification problems
    - neurons : int
        - number of initial neurons
    - max_neurons : int
        - max number of neurons
    - s_init : int
        - initial sigma for first neuron
    - eval_thresh : float
        - cutoff threshold for positive/negative classes
    - ifpart_thresh : float
        - threshold for if-part
    - err_delta : float
        - threshold for error criterion whether new neuron to be added
    - debug : debug flag

    Methods
    =======
    - build_model :
        - build and compile model
    - loss_function :
        - custom loss function per Leng, Prasad, McGinnity (2004)
    - train_model :
        - train on data
    - model_predictions :
        - yield model predictions without full evaluation
    - error_criterion :
        - considers generalized performance of overall network
        - add neuron if error above predefined error threshold (delta)
    - if_part_criterion :
        - checks if current fuzzy rules cover/cluster input vector suitably

    Secondary Methods
    =================
    - get_layer :
        - return layer object from model by name
    - get_layer_weights :
        - get current weights from any layer in model
    - get_layer_output :
        - get test output from any layer in model
    - min_dist_vector :
        - get min_dist_vector used when adding neurons
    - new_neuron_weights :
        - get weights for new neuron to be added
    - initialize_model :
        - initialize neuron weights if only 1 neuron
    """

    def __init__(self, X_train, X_test, y_train, y_test,    # data attributes
                 neurons=1, max_neurons=100, s_init=4,      # neuron initialization parameters
                 eval_thresh=0.5, ifpart_thresh=0.1354,     # evaluation and ifpart threshold
                 err_delta=0.12,                            # delta tolerance for errors
                 prob_type='classification',                # type of problem (classification/regression)
                 debug=True):

        # set debug flag
        self.__debug = debug

        # set output problem type
        if prob_type.lower() not in ['classification', 'regression']:
            raise ValueError("Invalid problem type")
        self.prob_type = prob_type

        # set data attributes
        # validate numpy arrays
        for data in [X_train, X_test, y_train, y_test]:
            if type(data) is not np.ndarray:
                raise ValueError("Input data must be NumPy arrays")

        # validate one-hot-encoded y values if classification
        if self.prob_type == 'classification':
            # convert to one-hot-encoding if y is one dimensional
            if y_test.ndim == 1:
                print('Converting y data to one-hot-encodings')
                # get number of samples in training data
                train_samples = y_train.shape[0]
                # convert complete y vector at once then split again
                y = np.concatenate([y_train, y_test])
                y = to_categorical(y)
                y_train = y[:train_samples]
                y_test = y[train_samples:]
            # set number of classes based on
            self.classes = y_test.shape[1]

        # set data attributes
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # set neuron attributes
        # initial number of neurons
        if type(neurons) is not int or neurons <= 0:
            raise ValueError("Must enter positive integer")
        self.neurons = neurons

        # max number of neurons
        if type(max_neurons) is not int or max_neurons < neurons:
            raise ValueError("Must enter positive integer no less than number of neurons")
        self.max_neurons = max_neurons

        # verify non-negative parameters
        non_neg_params = {'eval_thresh': eval_thresh,
                  'ifpart_thresh': ifpart_thresh,
                  'err_delta': err_delta}
        for param, val in non_neg_params.items():
            if val < 0:
                raise ValueError("Entered negative parameter: {}".format(param))

        # set calculation attributes
        self._eval_thresh = eval_thresh
        self._ifpart_thresh = ifpart_thresh
        self._err_delta = err_delta

        # build model and initialize if needed
        self.model = self.build_model()
        self.__initialize_model(s_init=s_init)

    def build_model(self, **kwargs):
        """
        Create and compile model
        - sets compiled model as self.model

        Layers
        ======
        1 - Input Layer
                input dataset
            - input shape  : (*, features)
        2 - Radial Basis Function Layer (Fuzzy Layer)
                layer to hold fuzzy rules for complex system
            - input : x
                shape: (*, features)
            - output : phi
                shape : (*, neurons)
        3 - Normalized Layer
                normalize each output of previous layer as
                relative amount from sum of all previous outputs
            - input : phi
                shape  : (*, neurons)
            - output : psi
                shape : (*, neurons)
        4 - Weighted Layer
                multiply bias vector (1+n_features, neurons) by
                parameter vector (1+n_features,) of parameters
                from each fuzzy rule
                multiply each product by output of each rule's
                layer from normalized layer
            - inputs : [x, psi]
                shape  : [(*, 1+features), (*, neurons)]
            - output : f
                shape : (*, neurons)
        5 - Output Layer
                summation of incoming signals from weighted layer
            - input shape  : (*, neurons)
            - output shape : (*,)
        """

        if self.__debug:
            print('BUILDING SOFNN WITH {} NEURONS'.format(self.neurons))

        # get shape of training data
        samples, feats = self.X_train.shape

        # add layers
        inputs = Input(name='Inputs', shape=(feats,))
        fuzz = FuzzyLayer(self.neurons)
        norm = NormalizedLayer(self.neurons)
        weights = WeightedLayer(self.neurons)
        raw = OutputLayer()

        # run through layers
        phi = fuzz(inputs)
        psi = norm(phi)
        f = weights([inputs, psi])
        raw_output = raw(f)
        final_out = raw_output
        # add softmax layer for classification problem
        if self.prob_type is 'classification':
            clasify = Dense(self.classes, activation='softmax')
            classes = clasify(raw_output)
            final_out = classes

        # TODO: determine logic for activation w/ regression
        # # extract activation from kwargs
        # if 'activation' not in kwargs:
        #     activation = 'sigmoid'
        # else:
        #     activation = kwargs['activation']
        # preds = Activation(name='OutputActivation', activation=activation)(raw_output)

        # define model
        model = Model(inputs=inputs, outputs=final_out)

        # default loss for classification
        if self.prob_type is 'classification':
            default_loss = self.loss_function
        # default loss for regression
        else:
            default_loss = 'mean_squared_error'
        loss = kwargs.get('loss', default_loss)

        # default optimizer for classification
        if self.prob_type is 'classification':
            default_optimizer = 'adam'
        # default optimizer for regression
        else:
            default_optimizer = 'rmsprop'
        optimizer = kwargs.get('optimizer', default_optimizer)

        # default metrics for classification
        if self.prob_type is 'classification':
            # default for binary classification
            if self.y_test.ndim == 2:
                default_metrics = ['binary_accuracy']
            # default for multi-class classification
            else:
                default_metrics = ['categorical_accuracy']
        # default metrics for regression
        else:
            default_metrics = ['accuracy']
        metrics = kwargs.get('metrics', default_metrics)

        # compile model and show model summary
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        if self.__debug:
            print(model.summary())

        return model

    @staticmethod
    def loss_function(y_true, y_pred):
        """
        Custom loss function

        E = exp{-sum[i=1,j; 1/2 * [pred(j) - test(j)]^2]}

        Parameters
        ==========
        y_true : np.array
            - true values
        y_pred : np.array
            - predicted values
        """
        return K.sum(1 / 2 * K.square(y_pred - y_true))

    def train_model(self, **kwargs):
        """
        Run currently saved model
        """
        if self.__debug:
            print('Training model...')

        # extract verbosity
        if 'verbose' not in kwargs:
            verbose = 1
        else:
            verbose = kwargs['verbose']
        # extract training epochs
        if 'epochs' not in kwargs:
            epochs = 100
        else:
            epochs = kwargs['epochs']
        # extract training batch size
        if 'batch_size' not in kwargs:
            batch_size = 32
        else:
            batch_size = kwargs['batch_size']

        # fit model and evaluate
        self.model.fit(self.X_train, self.y_train, verbose=verbose,
                       epochs=epochs, batch_size=batch_size, **kwargs)

    # TODO: update yields for predictions
    def model_predictions(self):
        """
        Evaluate currently trained model


        Returns
        =======
        y_pred : np.array
            - predicted values
            - shape: (samples,)
        """
        # get prediction values
        raw_pred = self.model.predict(self.X_test)
        y_pred = np.squeeze(np.where(raw_pred >= self._eval_thresh, 1, 0), axis=-1)
        return y_pred

    # TODO: validate logic
    def error_criterion(self, y_pred):
        """
        Check error criterion for neuron-adding process
            - return True if no need to grow neuron
            - return False if above threshold and need to add neuron

        Parameters
        ==========
        y_pred : np.array
            - model predictions
        """
        # mean of absolute test difference
        return mean_absolute_error(self.y_test, y_pred) <= self._err_delta

    # TODO: validate logic
    def if_part_criterion(self):
        """
        Check if-part criterion for neuron adding process
            - for each sample, get max of all neuron outputs (pre-normalization)
            - test whether max val at or above threshold
        """
        # get max val
        fuzz_out = self._get_layer_output('FuzzyRules')
        # check if max neuron output is above threshold
        maxes = np.max(fuzz_out, axis=-1) >= self._ifpart_thresh
        # return True if at least half of samples agree
        return (maxes.sum() / len(maxes)) >= 0.5

    def _get_layer(self, layer=None):
        """
        Get layer object based on input parameter
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get weights from
            - input can be layer name or index
        """
        # if named parameter
        if layer in [mlayer.name for mlayer in self.model.layers[1:]]:
            layer_out = self.model.get_layer(layer)
        # if indexed parameter
        elif layer in range(1, len(self.model.layers)):
            layer_out = self.model.layers[layer]
        else:
            raise ValueError('Error: layer must be layer name or index')
        return layer_out

    def _get_layer_weights(self, layer=None):
        """
        Get weights of layer based on input parameter
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get weights from
            - input can be layer name or index
        """
        return self._get_layer(layer).get_weights()

    def _get_layer_output(self, layer=None):
        """
        Get output of layer based on input parameter
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get test output from
            - input can be layer name or index
        """
        last_layer = self._get_layer(layer)
        intermediate_model = Model(inputs=self.model.input,
                                   outputs=last_layer.output)
        return intermediate_model.predict(self.X_test)

    # TODO: validate logic for numpy arrays
    def _min_dist_vector(self):
        """
        Get minimum distance vector

        Returns
        =======
        min_dist : np.array
            - average minimum distance vector across samples
            - shape: (features, neurons)
        """
        # get input values and fuzzy weights
        x = self.X_train.values
        samples = x.shape[0]
        c = self._get_layer_weights('FuzzyRules')[0]

        # align x and c and assert matching dims
        aligned_x = x.repeat(self.neurons). \
            reshape(x.shape + (self.neurons,))
        aligned_c = c.repeat(samples).reshape((samples,) + c.shape)
        assert aligned_x.shape == aligned_c.shape

        # average the minimum distance across samples
        return np.abs(aligned_x - aligned_c).mean(axis=0)

    # TODO: validate logic for numpy arrays
    def _new_neuron_weights(self, dist_thresh=1):
        """
        Return new c and s weights for k new fuzzy neuron

        Parameters
        ==========
        dist_thresh : float
            - multiplier of average features values to use as distance thresholds

        Returns
        =======
        ck : np.array
            - average minimum distance vector across samples
            - shape: (features,)
        sk : np.array
            - average minimum distance vector across samples
            - shape: (features,)
        """
        # get input values and fuzzy weights
        x = self.X_train.values
        c, s = self._get_layer_weights('FuzzyRules')

        # get minimum distance vector
        min_dist = self._min_dist_vector()
        # get minimum distance across neurons
        # and arg-min for neuron with lowest distance
        dist_vec = min_dist.min(axis=-1)
        min_neurs = min_dist.argmin(axis=-1)

        # get min c and s weights
        c_min = c[:, min_neurs].diagonal()
        s_min = s[:, min_neurs].diagonal()
        assert c_min.shape == s_min.shape

        # set threshold distance as factor of mean
        # value for each feature across samples
        kd_i = x.mean(axis=0) * dist_thresh

        # get final weight vectors
        ck = np.where(dist_vec <= kd_i, c_min, x.mean(axis=0))
        sk = np.where(dist_vec <= kd_i, s_min, dist_vec)
        return ck, sk

    def __initialize_model(self, s_init=4):
        """
        Randomly initialize neuron weights with random samples
        from X_train dataset

        Parameters
        ==========
        s_init : int
            - initial sigma value for all neuron centers
        """
        # c
        # set centers as random sampled index values
        samples = np.random.randint(0, len(self.X_train), self.neurons)
        x_i = np.array([self.X_train[samp] for samp in samples])
        # reshape to (features, neurons) from (neurons, features)
        c_init = x_i.T

        # s
        # repeat s_init value to array shaped like c_init
        s_init = np.repeat(s_init, c_init.size).reshape(c_init.shape)

        # set weights
        start_weights = [c_init, s_init]
        self._get_layer('FuzzyRules').set_weights(start_weights)
        # validate weights updated as expected
        final_weights = self._get_layer_weights('FuzzyRules')
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])
