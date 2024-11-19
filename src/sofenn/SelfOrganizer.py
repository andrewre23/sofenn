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

from keras.models import clone_model, Model

from sklearn.metrics import mean_absolute_error, mean_squared_error

# custom Fuzzy Layers
from .FuzzyNetwork import FuzzyNetwork
from .layers import FuzzyLayer, NormalizedLayer, WeightedLayer, OutputLayer


class SelfOrganizingFuzzyNN(object):
    """
    Self-Organizing Fuzzy Neural Network
    ====================================

    Organizer
    =========

    -Implemented per description in:
        "An on-line algorithm for creating self-organizing
        fuzzy neural networks" - Leng, Prasad, McGinnity (2004)
    -Composed of 5 layers with varying "fuzzy rule" nodes

    Attributes
    ==========
    - ksig : float
        - factor to widen centers
    - max_widens : int
        - max iterations for widening centers
    - prune_tol : float
        - tolerance limit for RMSE (0 < lambda < 1)
    - k_mae : float
        - expected RMSE for error when pruning neurons
    - debug : debug flag

    Methods
    =======
    - self_organize :
        - main method for network to learn optimal network structure
    - organize :
        - one iteration of logic to test network structure
    - build_network :
        - create and initialize FuzzyNetwork object
    - build_model :
        - build fuzzy network
    - compile_model :
        - compile fuzzy network
    - train_model :
        - train fuzzy network on currently set training data
    - recompile_model :
        - recompile already existing model after modifications
    - duplicate_model :
        - create copy of model for safely modifying original model
    - widen_centers :
        - widen centers of membership functions to better cluster the dataset
    - add_neuron :
        - add new neuron to network
    - new_neuron_weights :
        - yield neuron weights to use when adding new neuron
    - min_dist_vector :
        - calculate minimum distance vector for calculating new neuron weights
    - prune_neurons :
        - remove unnecessary neurons from network architecture
    - combine_membership_functions :
        - combine similar membership functions to simplify network
    """

    def __init__(self,
                 ksig=1.12, max_widens=250,  # adding neuron or widening centers
                 prune_tol=0.8, k_rmse=0.1,  # pruning parameters
                 debug=True):

        # set debug flag
        self.__debug = debug

        # create empty network and model attributes
        self.network = None
        self.model = None

        # value checks for input parameters

        # center-widening factor
        if ksig <= 1:
            raise ValueError('Widening factor (ksig) must be larger than 1')
        # max center widens
        if max_widens < 0:
            raise ValueError('Max center widens must be at least 0')
        # prune tolerance
        if not 0 < prune_tol < 1:
            raise ValueError('Prune tolerance must be between 0 and 1')
        # expected error
        if k_rmse <= 0:
            raise ValueError('Expected RMSE must be greater than 0')

        # set self-organizing attributes
        self._ksig = ksig
        self._max_widens = max_widens
        self._prune_tol = prune_tol
        self._k_rmse = k_rmse

    def self_organize(self, **kwargs):
        """
        Main run function to handle organization logic

        - Train initial model in parameters then begin self-organization
        - If fails If-Part test, widen rule widths
        - If still fails, reset to original widths
            then add neuron and retrain weights
        """

        # create simple alias for self.network
        fuzzy_net = self.network

        # initial training of model - yields predictions
        if self.__debug:
            print('Beginning model training...')
        self.train_model(**kwargs)

        # set organization iterations counter
        org_ints = 1

        # run update logic until passes criterion checks
        while not fuzzy_net.error_criterion() and not fuzzy_net.if_part_criterion():
            if self.__debug:
                print('Organization iteration {}...'.format(org_ints))

            # run criterion checks and organize accordingly
            self.organize(**kwargs)

            # quit if above max neurons allowed
            if fuzzy_net.neurons >= fuzzy_net.max_neurons:
                if self.__debug:
                    print('Maximum neurons reached')
                    print('Terminating self-organizing process')

            # increase counter
            org_ints += 1

        # print terminal message if successfully organized
        if self.__debug:
            print('Self-Organization complete!')
            print('If-Part Criterion and Error Criterion both satisfied')

    def organize(self, **kwargs):
        """
        Run one iteration of organizational logic
        - check on system error and if-part criteron
        - add neurons or prune if needed
        """

        # create simple alias for self.network
        fuzzy_net = self.network

        # get copy of initial fuzzy weights
        start_weights = fuzzy_net.get_layer_weights(1)

        # widen centers if necessary
        if not fuzzy_net.if_part_criterion():
            self.widen_centers()

        # add neuron if necessary
        if not fuzzy_net.error_criterion():
            # reset fuzzy weights if previously widened before adding
            curr_weights = fuzzy_net.get_layer_weights(1)
            if not np.array_equal(start_weights, curr_weights):
                fuzzy_net.get_layer(1).set_weights(start_weights)

            # add neuron and retrain model (if added)
            added = self.add_neuron(**kwargs)
            if added:
                self.train_model(**kwargs)

        # prune neurons and retrain model (if pruned)
        pruned = self.prune_neurons(**kwargs)
        if pruned:
            self.train_model(**kwargs)

        # check if needing to combine membership functions
        self.combine_membership_functions(**kwargs)

    def build_network(self,
                      X_train, X_test, y_train, y_test,           # data attributes
                      neurons=1, max_neurons=100,                 # neuron initialization parameters
                      ifpart_thresh=0.1354, ifpart_samples=0.75,  # ifpart threshold and percentage of samples needed
                      err_delta=0.12,                             # error criterion
                      prob_type='classification',                 # type of problem (classification/regression)
                      **kwargs):
        """
        Create FuzzyNetwork object and set network and model attributes

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
        - neurons : int
            - number of initial neurons
        - max_neurons : int
            - max number of neurons
        - ifpart_thresh : float
            - threshold for if-part
        - ifpart_samples : float
            - percent of samples needed to meet ifpart criterion
        - err_delta : float
            - threshold for error criterion whether new neuron to be added
        """

        # Fuzzy network as network attribute
        self.network = FuzzyNetwork(X_train, X_test, y_train, y_test,
                                    neurons=neurons, max_neurons=max_neurons,
                                    ifpart_thresh=ifpart_thresh, ifpart_samples=ifpart_samples,
                                    err_delta=err_delta, prob_type=prob_type,
                                    debug=self.__debug, **kwargs)
        # shortcut reference to network model
        self.model = self.network.model

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

        # pass parameters to network method for building model
        self.network.build_model(**kwargs)

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

        # pass parameters to network method
        self.network.compile_model(init_c=init_c, random=random,
                                   init_s=init_s, s_0=s_0, **kwargs)

    def train_model(self, **kwargs):
        """
        Fit model on current training data
        """

        # pass parameters to network method
        self.network.train_model(**kwargs)

    def duplicate_model(self):
        """
        Create duplicate model as FuzzyNetwork with identical weights
        """
        # create duplicate model and update weights
        dupe_mod = clone_model(self.model)
        dupe_mod.set_weights(self.model.get_weights())
        return dupe_mod

    def rebuild_model(self, new_weights, new_neurons, **kwargs):
        """
        Create updated FuzzyNetwork by adding or pruning neurons and updating to new weights
        """
        # get config from current model and update output_dim of neuron layers
        config = self.model.get_config()
        for layer in config['layers']:
            if 'output_dim' in layer['config']:
                layer['config']['output_dim'] = new_neurons

        # load new model from custom config data and load new weights
        custom_objects = {'FuzzyLayer': FuzzyLayer,
                          'NormalizedLayer': NormalizedLayer,
                          'WeightedLayer': WeightedLayer,
                          'OutputLayer': OutputLayer}
        new_model = Model.from_config(config, custom_objects=custom_objects)
        new_model.set_weights(new_weights)

        # recompile model based on current model parameters
        optimizer = kwargs.get('optimizer', self.model.optimizer)
        loss = kwargs.get('loss', self.model.loss)
        metrics = kwargs.get('metrics', self.model.metrics)
        new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

        # update neuron attribute
        self.network.neurons = new_neurons
        return new_model

    def widen_centers(self):
        """
        Widen center of neurons to better cover data
        """
        # print alert of successful widening
        if self.__debug:
            print('Widening centers...')

        # create simple alias for self.network
        fuzzy_net = self.network

        # get fuzzy layer and output to find max neuron output
        fuzz_layer = fuzzy_net.get_layer(1)

        # get old weights and create current weight vars
        c, s = fuzz_layer.get_weights()

        # repeat until if-part criterion satisfied
        # only perform for max iterations
        counter = 0
        while not fuzzy_net.if_part_criterion():

            counter += 1
            # check if max iterations exceeded
            if counter > self._max_widens:
                if self.__debug:
                    print('Max iterations reached ({})'
                          .format(counter - 1))
                return False

            # get neuron with max-output for each sample
            # then select the most common one to update
            fuzz_out = fuzzy_net.get_layer_output(1)
            maxes = np.argmax(fuzz_out, axis=-1)
            max_neuron = np.argmax(np.bincount(maxes.flat))

            # select minimum width to expand
            # and multiply by factor
            mf_min = s[:, max_neuron].argmin()
            s[mf_min, max_neuron] = self._ksig * s[mf_min, max_neuron]

            # update weights
            new_weights = [c, s]
            fuzz_layer.set_weights(new_weights)

        # print alert of successful widening
        if self.__debug:
            if counter == 0:
                print('Centers not widened')
            else:
                print('Centers widened after {} iterations'.format(counter))

    def add_neuron(self, **kwargs):
        """
        Add one additional neuron to the network
            - new FuzzyLayer  weights will be added using minimum distance vector calculation
            - new WeightedLayer weights are always a new column of 1
        """
        if self.__debug:
            print('Adding neuron...')

        # get current weights
        w = self.model.get_weights()
        c_curr, s_curr, a_curr = w[0], w[1], w[2]

        # get weights for new neuron
        ck, sk = self.new_neuron_weights()
        # expand dim for stacking
        ck = np.expand_dims(ck, axis=-1)
        sk = np.expand_dims(sk, axis=-1)
        c_new = np.hstack((c_curr, ck))
        s_new = np.hstack((s_curr, sk))

        # update a vector to include column of 1s
        a_add = np.ones((a_curr.shape[-2]))
        a_new = np.column_stack((a_curr, a_add))

        # update weights to include new neurons
        w[0], w[1], w[2] = c_new, s_new, a_new

        # update model and neurons
        self.network.model = self.rebuild_model(new_weights=w, new_neurons=self.network.neurons + 1, **kwargs)
        self.model = self.network.model

        if self.__debug:
            print('Neuron successfully added! - {} current neurons...'.format(self.network.neurons))
        return True

    def prune_neurons(self, **kwargs):
        """
        Prune any unimportant neurons per effect on RMSE
        """
        if self.__debug:
            print('Pruning neurons...')

        # create simple alias for self.network
        fuzzy_net = self.network

        # quit if only 1 neuron exists
        if fuzzy_net.neurons == 1:
            if self.__debug:
                print('Skipping pruning steps - only 1 neuron exists')
            return

        # get current training predictions
        preds = self.model.predict(fuzzy_net.X_train)

        # calculate mean-absolute-error on training data
        E_rmse = mean_squared_error(fuzzy_net.y_train, preds)

        # create duplicate model and get both sets of model weights
        prune_model = self.duplicate_model()
        act_weights = self.model.get_weights()

        # for each neuron, zero it out in prune model
        # and get change in mae for dropping neuron
        delta_E = []
        for neuron in range(fuzzy_net.neurons):
            # reset prune model weights to actual weights
            prune_model.set_weights(act_weights)

            # get current prune weights
            w = prune_model.get_weights()
            # zero our i neuron column in weighted vector
            a = w[2]
            a[:, neuron] = 0
            prune_model.set_weights(w)

            # predict values with new zeroed out weights
            neur_pred = prune_model.predict(fuzzy_net.X_train)
            neur_rmae = mean_absolute_error(fuzzy_net.y_train, neur_pred)

            # append difference in rmse and new prediction rmse
            delta_E.append(neur_rmae - E_rmse)

        # convert delta_E to numpy array
        delta_E = np.array(delta_E)
        # choose max of tolerance or threshold limit
        E = max(self._prune_tol * E_rmse, self._k_rmse)

        # iterate over each neuron in ascending importance
        # and prune until hit "important" neuron
        deleted = []
        # for each neuron up to second most important
        for neuron in delta_E.argsort()[:-1]:
            # reset prune model weights to actual weights
            prune_model.set_weights(act_weights)

            # get current prune weights
            w = prune_model.get_weights()
            a = w[2]
            # zero out previous deleted neurons
            for delete in deleted:
                a[:, delete] = 0
            prune_model.set_weights(w)

            # predict values with new zeroed out weights
            neur_pred = prune_model.predict(fuzzy_net.X_train)
            E_rmae_del = mean_absolute_error(fuzzy_net.y_train, neur_pred)

            # if E_mae_del < E
            # delete neuron
            if E_rmae_del < E:
                deleted.append(neuron)
                continue
            # quit deleting if >= E
            else:
                break

        # exit if no neurons to be deleted
        if not deleted:
            if self.__debug:
                print('No neurons detected for pruning')
            return False
        else:
            if self.__debug:
                print('Neurons to be deleted: ')
                print(deleted)

        # reset prune model weights to actual weights
        prune_model.set_weights(act_weights)
        # get current prune weights and remove deleted neurons
        w = prune_model.get_weights()
        for i, weight in enumerate(w[:3]):
            w[i] = np.delete(weight, deleted, axis=-1)

        # update model with updated weights
        self.network.model = self.rebuild_model(new_weights=w, new_neurons=self.network.neurons - len(deleted),
                                                **kwargs)
        self.model = self.network.model

        if self.__debug:
            print('{} neurons successfully pruned! - {} current neurons...'.
                  format(len(deleted), self.network.neurons))
        return True

    def new_neuron_weights(self, dist_thresh=1):
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

        # create simple alias for self.network
        fuzzy_net = self.network

        # get input values and fuzzy weights
        x = fuzzy_net.X_train
        c, s = fuzzy_net.get_layer_weights(1)

        # get minimum distance vector
        min_dist = self.min_dist_vector()
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

    def min_dist_vector(self):
        """
        Get minimum distance vector

        Returns
        =======
        min_dist : np.array
            - average minimum distance vector across samples
            - shape: (features, neurons)
        """

        # create simple alias for self.network
        fuzzy_net = self.network

        # get input values and fuzzy weights
        x = fuzzy_net.X_train
        samples = x.shape[0]
        c, s = fuzzy_net.get_layer_weights(1)

        # align x and c and assert matching dims
        aligned_x = x.repeat(fuzzy_net.neurons). \
            reshape(x.shape + (fuzzy_net.neurons,))
        aligned_c = c.repeat(samples).reshape((samples,) + c.shape)
        assert aligned_x.shape == aligned_c.shape

        # average the minimum distance across samples
        return np.abs(aligned_x - aligned_c).mean(axis=0)

    # TODO: add method combining membership functions
    def combine_membership_functions(self, **kwargs):
        """
        Function to combine redundant membership functions to simplify training parameters
        """
        pass
