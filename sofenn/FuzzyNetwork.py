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
    - ifpart_thresh : float
        - threshold for if-part
    - err_delta : float
        - threshold for error criterion whether new neuron to be added
    - debug : debug flag

    Methods
    =======
    - build_model :
        - build Fuzzy Network and set as model attribute
    - compile_model :
        - compile Fuzzy Network
    - loss_function :
        - custom loss function per Leng, Prasad, McGinnity (2004)
    - train_model :
        - train on data
    - model_predictions :
        - yield model predictions without full evaluation
    - model_evaluations :
        - evaluate models and yield metrics
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
        
    Protected Methods
    =================
    - initialize_centers :
        - initialize neuron centers
    - initialize_widths :
        - initialize neuron weights based on parameter
    """
    def __init__(self, X_train, X_test, y_train, y_test,    # data attributes
                 neurons=1, max_neurons=100,                # neuron initialization parameters
                 ifpart_thresh=0.1354, err_delta=0.12,      # ifpart and error thresholds
                 prob_type='classification',                # type of problem (classification/regression)
                 debug=True, **kwargs):

        # set debug flag
        self._debug = debug

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
        non_neg_params = {'ifpart_thresh': ifpart_thresh,
                          'err_delta': err_delta}
        for param, val in non_neg_params.items():
            if val < 0:
                raise ValueError("Entered negative parameter: {}".format(param))

        # set calculation attributes
        self._ifpart_thresh = ifpart_thresh
        self._err_delta = err_delta

        # define model and set model attribute
        self.model = None
        self.build_model(**kwargs)

    def build_model(self, **kwargs):
        """
        Build and initialize Model if needed

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
        if self._debug:
            print('Building Fuzzy Network with {} neurons...'
                  .format(self.neurons))

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
            clasify = Dense(self.classes,
                            name='Softmax', activation='softmax')
            classes = clasify(raw_output)
            final_out = classes

        # TODO: determine logic for activation w/ regression
        # # extract activation from kwargs
        # if 'activation' not in kwargs:
        #     activation = 'sigmoid'
        # else:
        #     activation = kwargs['activation']
        # preds = Activation(name='OutputActivation', activation=activation)(raw_output)

        # remove name from kwargs
        if 'name' in kwargs:
            kwargs.pop('name')

        # define model and set as model attribute
        model = Model(inputs=inputs, outputs=final_out,
                      name='FuzzyNetwork', **kwargs)
        self.model = model

        if self._debug:
            print('...Model successfully built!')

    def compile_model(self, init_c=True, random=True, init_s=True, s_0=4.0, **kwargs):
        """
        Create and compile model
        - sets compiled model as self.model

        Parameters
        ==========
        init_c : bool
            - run method to initialize centers or take default initializations
        random : bool
            - take either random samples or first samples that appear in training data
        init_s : bool
            - run method to initialize widths or take default initializations
        s_0 : float
            - value for initial centers of neurons
        """
        if self._debug:
            print('Compiling model...')

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
        self.model.compile(loss=loss, optimizer=optimizer,
                           metrics=metrics, **kwargs)

        # initialize fuzzy rule centers
        if init_c:
            self._initialize_centers(random=random)

        # initialize fuzzy rule widths
        if init_s:
            self._initialize_widths(s_0=s_0)

        # print model summary
        if self._debug:
            print(self.model.summary())

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
        Fit model on current training data
        """
        if self._debug:
            print('Training model...')

        # default verbosity
        if 'verbose' not in kwargs:
            kwargs['verbose'] = 1

        # default training epochs
        if 'epochs' not in kwargs:
            kwargs['epochs'] = 250

        # default training batch size
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32

        # fit model to dataset
        self.model.fit(self.X_train, self.y_train, **kwargs)

    def model_predictions(self):
        """
        Evaluate currently trained model and return predictions


        Returns
        =======
        preds : np.array
            - predicted values
            - shape: (samples,) or (samples, classes)
        """
        # get prediction values
        preds = self.model.predict(self.X_test)
        return preds

    def model_evaluation(self):
        """
        Evaluate current test dataset on model
        """

        # run model evaluation
        self.model.evaluate(self.X_test, self.y_test)

    def error_criterion(self):
        """
        Check error criterion for neuron-adding process
            - considers generalization performance of model

        Returns
        =======
        - True:
            if criteron satisfied and no need to grow neuron
        - False:
            if criteron not met and need to add neuron
        """
        # mean of absolute test difference
        y_pred = self.model_predictions()
        return mean_absolute_error(self.y_test, y_pred) <= self._err_delta

    def if_part_criterion(self, thresh=0.75):
        """
        Check if-part criterion for neuron-adding process
            - considers whether current fuzzy rules suitably cover inputs

            - for each sample, get max of all neuron outputs (pre-normalization)
            - test whether max val at or above threshold
            - overall criterion met if criterion met for 'thresh'% of samples

        Returns
        =======
        - True:
            if criteron satisfied and no need to widen centers
        - False:
            if criteron not met and need to widen neuron centers
        """
        # validate value of threshold parameter
        if not 0 < thresh <= 1.0:
            raise ValueError('Threshold must be between 0 and 1')

        # get max val
        fuzz_out = self.get_layer_output('FuzzyRules')
        # check if max neuron output is above threshold
        maxes = np.max(fuzz_out, axis=-1) >= self._ifpart_thresh
        # return True if at least half of samples agree
        return (maxes.sum() / len(maxes)) >= thresh

    def get_layer(self, layer=None):
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

    def get_layer_weights(self, layer=None):
        """
        Get weights of layer based on input parameter
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get weights from
            - input can be layer name or index
        """
        return self.get_layer(layer).get_weights()

    def get_layer_output(self, layer=None):
        """
        Get output of layer based on input parameter
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get test output from
            - input can be layer name or index
        """
        # create prediction from intermediate model ending at desired layer
        last_layer = self.get_layer(layer)
        intermediate_model = Model(inputs=self.model.input,
                                   outputs=last_layer.output)
        return intermediate_model.predict(self.X_test)

    def _initialize_centers(self, random=True):
        """
        Initialize neuron center weights with samples
        from X_train dataset

        Parameters
        ==========
        random: bool
            - take random samples from training data or
            take first n instances (n=# of neurons)
        """
        if random:
            # set centers as random sampled index values
            samples = np.random.randint(0, len(self.X_train), self.neurons)
            x_i = np.array([self.X_train[samp] for samp in samples])
        else:
            # take first few samples, one for each neuron
            x_i = self.X_train[:self.neurons]
        # reshape from (neurons, features) to (features, neurons)
        c_init = x_i.T

        # set weights
        c, s = self.get_layer_weights('FuzzyRules')
        start_weights = [c_init, s]
        self.get_layer('FuzzyRules').set_weights(start_weights)
        # validate weights updated as expected
        final_weights = self.get_layer_weights('FuzzyRules')
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])

    def _initialize_widths(self, s_0=4.0):
        """
        Initialize neuron widths

        Parameters
        ==========
        s_0 : float
            - initial sigma value for all neuron centers
        """
        # get current center and width weights
        c, s = self.get_layer_weights('FuzzyRules')

        # repeat s_0 value to array shaped like s
        s_init = np.repeat(s_0, s.size).reshape(s.shape)

        # set weights
        start_weights = [c, s_init]
        self.get_layer('FuzzyRules').set_weights(start_weights)
        # validate weights updated as expected
        final_weights = self.get_layer_weights('FuzzyRules')
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])
