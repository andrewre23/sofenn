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
# import pandas as pd
# import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Activation

from sklearn.metrics import confusion_matrix, classification_report, \
    mean_absolute_error, roc_auc_score

# custom Fuzzy Layers
from .layers import FuzzyLayer, NormalizedLayer, WeightedLayer, OutputLayer
from .FuzzyNetwork import FuzzyNetwork


class SelfOrganizer(object):
    """
    Self-Organizing Fuzzy Neural Network
    ====================================

    Organizer
    =========

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
    # TODO: remove defaults set in Fuzzy Network
    def __init__(self, X_train, X_test, y_train, y_test,     # data attributes
                 neurons=1, s_init=4, max_neurons=100,       # initialization parameters
                 epochs=250, batch_size=None,                  # training data
                 eval_thresh=0.5, ifpart_thresh=0.1354,      # evaluation and ifpart threshold
                 ksig=1.12, max_widens=250, err_delta=0.12,  # adding neuron or widening centers
                 prune_tol=0.85, k_mae=0.1,                  # pruning parameters
                 debug=True):
        # set debug flag
        self.__debug = debug

        # set data attributes
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        # set initial number of neurons
        self.__neurons = neurons

        # set remaining attributes
        self._max_neurons = max_neurons
        self._epochs = epochs
        self._batch_size = batch_size
        self._eval_thresh = eval_thresh
        self._ifpart_thresh = ifpart_thresh
        self._ksig = ksig
        self._max_widens = max_widens
        self._delta = err_delta
        self._prune_tol = prune_tol
        self._k_mae = k_mae

        # TODO: add fuzzy network attribute initialization
        # build model and initialize if needed
        self.model = self.build_model()
        if self.__neurons == 1:
            self.__initialize_model(s_init=s_init)

    # TODO: remove
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
            print('\nBUILDING SOFNN WITH {} NEURONS'.format(self.__neurons))

        # get shape of training data
        samples, feats = self._X_train.shape

        # add layers
        inputs = Input(name='Inputs', shape=(feats,))
        fuzz = FuzzyLayer(self.__neurons)
        norm = NormalizedLayer(self.__neurons)
        weights = WeightedLayer(self.__neurons)
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

    # TODO: validate logic and update references
    def self_organize(self):
        """
        Main run function to handle organization logic

        - Train initial model in parameters then begin self-organization
        - If fails If-Part test, widen rule widths
        - If still fails, reset to original widths
            then add neuron and retrain weights
        """
        # initial training of model - yields predictions
        if self.__debug:
            print('Beginning model training...')
        self._train_model()
        if self.__debug:
            print('Initial Model Evaluation')
        y_pred = self._evaluate_model(eval_thresh=self._eval_thresh)

        # run update logic until passes criterion checks
        while not self.error_criterion(y_pred) and \
                not self.if_part_criterion():
            # run criterion checks and organize accordingly
            self.organize(y_pred=y_pred)

            # quit if above max neurons allowed
            if self.__neurons >= self._max_neurons:
                if self.__debug:
                    print('\nMaximum neurons reached')
                    print('Terminating self-organizing process')
                    print('\nFinal Evaluation')
                    self._evaluate_model(eval_thresh=self._eval_thresh)

            # update predictions
            y_pred = self._evaluate_model(eval_thresh=self._eval_thresh)

        # print terminal message if successfully organized
        if self.__debug:
            print('\nSelf-Organization complete!')
            print('If-Part and Error Criterion satisfied')
            print('\nFinal Evaluation')
            self._evaluate_model(eval_thresh=self._eval_thresh)

    # TODO: validate logic and update references
    def organize(self, y_pred):
        """
        Run one iteration of organizational logic
        - check on system error and if-part criteron
        - add neurons or prune if needed

        Parameters
        ==========
        y_pred : np.array
            - predictions
        """

        # get copy of initial fuzzy weights
        start_weights = self._get_layer_weights('FuzzyRules')

        # widen centers if necessary
        if not self.if_part_criterion():
            self.widen_centers()

        # add neuron if necessary
        if not self.error_criterion(y_pred=y_pred):
            # reset fuzzy weights if previously widened before adding
            if not np.array_equal(start_weights, self._get_layer_weights('FuzzyRules')):
                self._get_layer('FuzzyRules').set_weights(start_weights)
            # add neuron and retrain model
            self.add_neuron()
            self._train_model()

        # updated prediction and prune neurons
        y_pred_new = self._model_predictions()
        self.prune_neurons(y_pred=y_pred_new)

    # TODO: validate logic and update references
    def add_neuron(self):
        """
        Add extra neuron to model while
        keeping current neuron weights
        """
        if self.__debug:
            print('\nAdding neuron...')

        # get current weights
        c_curr, s_curr = self._get_layer_weights('FuzzyRules')

        # get weights for new neuron
        ck, sk = self._new_neuron_weights()
        # expand dim for stacking
        ck = np.expand_dims(ck, axis=-1)
        sk = np.expand_dims(sk, axis=-1)
        c_new = np.hstack((c_curr, ck))
        s_new = np.hstack((s_curr, sk))

        # increase neurons and rebuild model
        self.__neurons += 1
        self.model = self.build_model()

        # update weights
        new_weights = [c_new, s_new]
        self._get_layer('FuzzyRules').set_weights(new_weights)

        # validate weights updated as expected
        final_weights = self._get_layer_weights('FuzzyRules')
        assert np.allclose(c_new, final_weights[0], 1e-3)
        assert np.allclose(s_new, final_weights[1], 1e-3)

        # retrain model since new neuron added
        self._train_model()

    # TODO: validate logic and update references
    def prune_neurons(self, y_pred):
        """
        Prune any unimportant neurons per effect on RMSE

        Parameters
        ==========
        y_pred : np.array
            - predicted values
        """
        if self.__debug:
            print('\nPruning neurons...')

        # quit if only 1 neuron exists
        if self.__neurons == 1:
            if self.__debug:
                print('Skipping pruning steps - only 1 neuron exists')
            return

        # calculate mean-absolute-error
        E_rmae = mean_absolute_error(self._y_test.values, y_pred)

        # create duplicate model and get both sets of model weights
        prune_model = self.build_model(False)
        act_weights = self.model.get_weights()

        # for each neuron, zero it out in prune model
        # and get change in mae for dropping neuron
        delta_E = []
        for neur in range(self.__neurons):
            # reset prune model weights to actual weights
            prune_model.set_weights(act_weights)

            # get current prune weights
            c, s, a = prune_model.get_weights()
            # zero our i neuron column in weight vector
            a[:, neur] = 0
            prune_model.set_weights([c, s, a])

            # predict values with new zeroed out weights
            neur_pred = prune_model.predict(self._X_test)
            y_pred_neur = np.squeeze(np.where(neur_pred >= self._eval_thresh, 1, 0), axis=-1)
            neur_rmae = mean_absolute_error(self._y_test.values, y_pred_neur)

            # append difference in rmse and new prediction rmse
            delta_E.append(neur_rmae - E_rmae)

        # convert delta_E to numpy array
        delta_E = np.array(delta_E)
        # choose max of tolerance or threshold limit
        E = max(self._prune_tol * E_rmae, self._k_mae)

        # iterate over each neuron in ascending importance
        # and prune until hit "important" neuron
        deleted = []
        # for each neuron up to second most important
        for neur in delta_E.argsort()[:-1]:
            # reset prune model weights to actual weights
            prune_model.set_weights(act_weights)

            # get current prune weights
            c, s, a = prune_model.get_weights()
            # zero out previous deleted neurons
            for delete in deleted:
                a[:, delete] = 0
            # zero our i neuron column in weight vector
            a[:, neur] = 0
            prune_model.set_weights([c, s, a])

            # predict values with new zeroed out weights
            neur_pred = prune_model.predict(self._X_test)
            y_pred_neur = np.squeeze(np.where(neur_pred >= self._eval_thresh, 1, 0), axis=-1)
            E_rmae_del = mean_absolute_error(self._y_test.values, y_pred_neur)

            # if E_mae_del < E
            # delete neuron
            if E_rmae_del < E:
                deleted.append(neur)
                continue
            # quit deleting if >= E
            else:
                break

        # exit if no neurons to be deleted
        if not deleted:
            if self.__debug:
                print('No neurons detected for pruning')
            return
        else:
            if self.__debug:
                print('Neurons to be deleted: ')
                print(deleted)

        # reset prune model weights to actual weights
        prune_model.set_weights(act_weights)
        # get current prune weights
        c, s, a = prune_model.get_weights()
        # delete prescribed neurons
        c = np.delete(c, deleted, axis=-1)
        s = np.delete(s, deleted, axis=-1)
        a = np.delete(a, deleted, axis=-1)

        # update neuron count and create new model with updated weights
        self.__neurons -= len(deleted)
        self.model = self.build_model(False)
        self.model.set_weights([c, s, a])

    # TODO: validate logic and update references
    def widen_centers(self):
        """
        Widen center of neurons to better cover data
        """
        # print alert of successful widening
        if self.__debug:
            print('\nWidening centers...')

        # get fuzzy layer and output to find max neuron output
        fuzz = self._get_layer('FuzzyRules')

        # get old weights and create current weight vars
        c, s = fuzz.get_weights()

        # repeat until if-part criterion satisfied
        # only perform for max iterations
        counter = 0
        while not self.if_part_criterion():

            counter += 1
            # check if max iterations exceeded
            if counter > self._max_widens:
                if self.__debug:
                    print('Max iterations reached ({})'
                          .format(counter - 1))
                return False

            # get neuron with max-output for each sample
            # then select the most common one to update
            fuzz_out = self._get_layer_output('FuzzyRules')
            maxes = np.argmax(fuzz_out, axis=-1)
            max_neuron = np.argmax(np.bincount(maxes.flat))

            # select minimum width to expand
            # and multiply by factor
            mf_min = s[:, max_neuron].argmin()
            s[mf_min, max_neuron] = self._ksig * s[mf_min, max_neuron]

            # update weights
            new_weights = [c, s]
            fuzz.set_weights(new_weights)

        # print alert of successful widening
        if self.__debug:
            print('Centers widened after {} iterations'.format(counter))

    # TODO: remove - redundant
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
        return mean_absolute_error(self._y_test, y_pred) <= self._delta

    # TODO: remove - redundant
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

    # TODO: remove - redundant
    def __initialize_model(self, s_init=4):
        """
        Initialize neuron weights

        c_init = Average(X).T
        s_init = s_init

        """
        # derive initial c and s
        # set initial center as first training value
        x_i = self._X_train.values[0]
        c_init = np.expand_dims(x_i, axis=-1)
        s_init = np.repeat(s_init, c_init.size).reshape(c_init.shape)
        start_weights = [c_init, s_init]
        self._get_layer('FuzzyRules').set_weights(start_weights)

        # validate weights updated as expected
        final_weights = self._get_layer_weights('FuzzyRules')
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])

    # TODO: remove - redundant
    def _train_model(self):
        """
        Run currently saved model
        """
        # fit model and evaluate
        self.model.fit(self._X_train, self._y_train, verbose=0,
                       epochs=self._epochs, batch_size=self._batch_size)

    # TODO: remove - redundant
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
        raw_pred = self.model.predict(self._X_test)
        y_pred = np.squeeze(np.where(raw_pred >= self._eval_thresh, 1, 0), axis=-1)
        return y_pred

    # TODO: add logic to demo notebook
    # def _evaluate_model(self, eval_thresh=0.5):
    #     """
    #     Evaluate currently trained model
    #
    #     Parameters
    #     ==========
    #     eval_thresh : float
    #         - cutoff threshold for positive/negative classes
    #
    #     Returns
    #     =======
    #     y_pred : np.array
    #         - predicted values
    #         - shape: (samples,)
    #     """
    #     # calculate accuracy scores
    #     scores = self.model.evaluate(self._X_test, self._y_test, verbose=1)
    #     raw_pred = self.model.predict(self._X_test)
    #     y_pred = np.squeeze(np.where(raw_pred >= eval_thresh, 1, 0), axis=-1)
    #
    #     # get prediction scores and prediction
    #     accuracy = scores[1]
    #     auc = roc_auc_score(self._y_test, raw_pred)
    #     mae = mean_absolute_error(self._y_test, y_pred)
    #
    #     # print accuracy and AUC score
    #     print('\nAccuracy Measures')
    #     print('=' * 21)
    #     print("Accuracy:  {:.2f}%".format(100 * accuracy))
    #     print("MAPE:      {:.2f}%".format(100 * mae))
    #     print("AUC Score: {:.2f}%".format(100 * auc))
    #
    #     # print confusion matrix
    #     print('\nConfusion Matrix')
    #     print('=' * 21)
    #     print(pd.DataFrame(confusion_matrix(self._y_test, y_pred),
    #                        index=['true:no', 'true:yes'], columns=['pred:no', 'pred:yes']))
    #
    #     # print classification report
    #     print('\nClassification Report')
    #     print('=' * 21)
    #     print(classification_report(self._y_test, y_pred, labels=[0, 1]))
    #
    #     self._plot_results(y_pred=y_pred)
    #     # return predicted values
    #     return y_pred

    # TODO: remove - redundant
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

    # TODO: remove - redundant
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

    # TODO: remove - redundant
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
        return intermediate_model.predict(self._X_test)

    # TODO: validate logic and update references
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
        x = self._X_train.values
        samples = x.shape[0]
        c = self._get_layer_weights('FuzzyRules')[0]

        # align x and c and assert matching dims
        aligned_x = x.repeat(self.__neurons). \
            reshape(x.shape + (self.__neurons,))
        aligned_c = c.repeat(samples).reshape((samples,) + c.shape)
        assert aligned_x.shape == aligned_c.shape

        # average the minimum distance across samples
        return np.abs(aligned_x - aligned_c).mean(axis=0)

    # TODO: validate logic and update references
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
        x = self._X_train.values
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

    # TODO: add logic to demo notebook
    # def _plot_results(self, y_pred):
    #     """
    #     Plot predictions against time series
    #
    #     Parameters
    #     ==========
    #     y_pred : np.array
    #         - predicted values
    #     """
    #     # plotting results
    #     df_plot = pd.DataFrame()
    #
    #     # create pred/true time series
    #     df_plot['price'] = self._X_test['bitcoin_close']
    #     df_plot['pred'] = y_pred * df_plot['price']
    #     df_plot['true'] = self._y_test * df_plot['price']
    #     df_plot['hits'] = df_plot['price'] * (df_plot['pred'] == df_plot['true'])
    #     df_plot['miss'] = df_plot['price'] * (df_plot['pred'] != df_plot['true'])
    #
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     plt.plot(df_plot['price'], color='b')
    #     plt.bar(df_plot['price'].index, df_plot['hits'], color='g')
    #     plt.bar(df_plot['price'].index, df_plot['miss'], color='r')
    #     for label in ax.xaxis.get_ticklabels()[::400]:
    #         label.set_visible(False)
    #
    #     plt.title('BTC Close Price Against Predictions')
    #     plt.xlabel('Dates')
    #     plt.ylabel('BTC Price ($)')
    #     plt.grid(True)
    #     plt.xticks(df_plot['price'].index[::4],
    #                df_plot['price'].index[::4], rotation=70)
    #     plt.show()

    # TODO: remove - redundant
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
