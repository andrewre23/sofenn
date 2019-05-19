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
import pandas as pd

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Activation

from sklearn.metrics import confusion_matrix, classification_report, \
    mean_absolute_error, roc_auc_score

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
    - neurons : int
        - number of initial neurons
    - s_init : int
        - initial sigma for first neuron
    - epochs : int
        - training epochs
    - batch_size : int
        - training batch size
    - eval_thresh : float
        - cutoff for 0/1 class
    - ifpart_thresh : float
        - threshold for if-part
    - ksig : float
        - factor to widen centers
    - max_widens : int
        - max iterations for widening centers
    - delta : float
        - threshold for error criterion whether new neuron to be added
    - eval_thresh : float
        - cutoff threshold for positive/negative classes
    - prune_tol : float
        - tolerance limit for RMSE (0 < lambda < 1)
    - debug : debug flag

    Methods
    =======
    - build_model :
        - build and compile model
    - self_organize :
        - run main logic to organize FNN
    - error_criterion :
        - considers generalized performance of overall network
        - add neuron if error above predefined error threshold (delta)
    - if_part_criterion :
        - checks if current fuzzy rules cover/cluster input vector suitably
    - add_neuron :
        - add one neuron to model
    - prune_neuron :
        - remove neuron from model
    - combine_membership_functions :
        - combine similar membership functions

    Secondary Methods
    =================
    - initialize_model :
        - initialize neuron weights if only 1 neuron
    - train_model :
        - train on data
    - model_predictions :
        - yield model predictions without full evaluation
    - evaluate_model :
        - full evaluation of model on test data
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
    - loss_function :
        - custom loss function per Leng, Prasad, McGinnity (2004)
    """

    def __init__(self, X_train, X_test, y_train, y_test,     # data attributes
                 neurons=1, max_neurons=100, s_init=4,       # neuron initialization parameters
                 eval_thresh=0.5, ifpart_thresh=0.1354,      # evaluation and ifpart threshold
                 err_delta=0.12,
                 debug=True):
        # set debug flag
        self.__debug = debug

        # set data attributes
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # set neuron attributes
        self.neurons = neurons
        self._max_neurons = max_neurons

        # set remaining attributes
        self._eval_thresh = eval_thresh
        self._ifpart_thresh = ifpart_thresh
        self._err_delta = err_delta

        # build model and initialize if needed
        self.model = self.build_model()
        if self.neurons == 1:
            self.__initialize_model(s_init=s_init)

    def build_model(self, debug=True):
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
                shape: (*, features * neurons)
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

        if debug:
            print('\nBUILDING SOFNN WITH {} NEURONS'.format(self.neurons))

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
        preds = Activation(name='OutputActivation', activation='sigmoid')(raw_output)

        # compile model and output summary
        model = Model(inputs=inputs, outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'mape'])
        if debug:
            print(model.summary())

        return model

    def error_criterion(self, y_pred):
        """
        Check error criterion for neuron-adding process
            - return True if no need to grow neuron
            - return False if above threshold and need to add neuron

        Parameters
        ==========
        y_pred : np.array
            - predictions
        """
        # mean of absolute test difference
        return mean_absolute_error(self.y_test, y_pred) <= self._err_delta

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

    def __initialize_model(self, s_init=4):
        """
        Initialize neuron weights

        c_init = Average(X).T
        s_init = s_init

        """
        # derive initial c and s
        # set initial center as first training value
        x_i = self.X_train.values[0]
        c_init = np.expand_dims(x_i, axis=-1)
        s_init = np.repeat(s_init, c_init.size).reshape(c_init.shape)
        start_weights = [c_init, s_init]
        self._get_layer('FuzzyRules').set_weights(start_weights)

        # validate weights updated as expected
        final_weights = self._get_layer_weights('FuzzyRules')
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])

    def _train_model(self):
        """
        Run currently saved model
        """
        # fit model and evaluate
        self.model.fit(self.X_train, self.y_train, verbose=0,
                       epochs=self._epochs, batch_size=self._batch_size)

    def _model_predictions(self):
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

    def _evaluate_model(self, eval_thresh=0.5):
        """
        Evaluate currently trained model

        Parameters
        ==========
        eval_thresh : float
            - cutoff threshold for positive/negative classes

        Returns
        =======
        y_pred : np.array
            - predicted values
            - shape: (samples,)
        """
        # calculate accuracy scores
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        raw_pred = self.model.predict(self.X_test)
        y_pred = np.squeeze(np.where(raw_pred >= eval_thresh, 1, 0), axis=-1)

        # get prediction scores and prediction
        accuracy = scores[1]
        auc = roc_auc_score(self.y_test, raw_pred)
        mae = mean_absolute_error(self.y_test, y_pred)

        # print accuracy and AUC score
        print('\nAccuracy Measures')
        print('=' * 21)
        print("Accuracy:  {:.2f}%".format(100 * accuracy))
        print("MAPE:      {:.2f}%".format(100 * mae))
        print("AUC Score: {:.2f}%".format(100 * auc))

        # print confusion matrix
        print('\nConfusion Matrix')
        print('=' * 21)
        print(pd.DataFrame(confusion_matrix(self.y_test, y_pred),
                           index=['true:no', 'true:yes'], columns=['pred:no', 'pred:yes']))

        # print classification report
        print('\nClassification Report')
        print('=' * 21)
        print(classification_report(self.y_test, y_pred, labels=[0, 1]))

        # return predicted values
        return y_pred

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

    @staticmethod
    def _loss_function(y_true, y_pred):
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
