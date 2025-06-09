from typing import Tuple, Optional
import inspect

import numpy as np
from keras.api.models import clone_model, Model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sofenn.FuzzyNetwork import FuzzyNetwork
from sofenn.layers import FuzzyLayer, NormalizeLayer, WeightedLayer, OutputLayer


class FuzzySelfOrganizer(object):
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
                 model: Optional[FuzzyNetwork] = None,

                 max_neurons: int = 100,            # maximum neurons during organizing
                 ifpart_threshold: float = 0.1354,  # if-part threshold
                 ifpart_samples: float = 0.95,      # percent of samples needed above if-part threshold
                 err_delta: float = 0.12,           # error delta

                 k_sig: float = 1.12,               # TODO: add definition
                 max_widens: float = 250,           # adding neuron or widening centers
                 prune_tol: float = 0.8,            # pruning parameters
                 k_rmse:float = 0.1,                # TODO: add definition

                 **kwargs
                 ):

        # max number of neurons
        init_neurons = kwargs.get('neurons', inspect.signature(FuzzyNetwork).parameters['neurons'].default)
        if max_neurons < init_neurons:
            raise ValueError(f"Maximum neurons cannot be smaller than initialized neurons: {init_neurons}.")
        self.max_neurons = max_neurons

        # set calculation attributes
        if ifpart_threshold < 0:
            raise ValueError(f"If-Part Threshold must not be negative: {ifpart_threshold}")
        self.ifpart_thresh = ifpart_threshold

        if not 0 < ifpart_samples <= 1.0:
            raise ValueError(f'If-Part Samples must be between between 0 and 1: {ifpart_samples}')
        self.ifpart_samples = ifpart_samples

        if err_delta < 0:
            raise ValueError(f"Error Delta must not be negative: {err_delta}")
        self.err_delta = err_delta

        # center-widening factor
        if k_sig <= 1:
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
        self.ksig = k_sig
        self.max_widens = max_widens
        self.prune_tol = prune_tol
        self.k_rmse = k_rmse

        self.model = FuzzyNetwork(**kwargs) if model is None else model

    # def self_organize(self, **kwargs) -> None:
    #     """
    #     Main run function to handle organization logic
    #
    #     - Train initial model in parameters then begin self-organization
    #     - If fails If-Part test, widen rule widths
    #     - If still fails, reset to original widths
    #         then add neuron and retrain weights
    #     """
    #     # initial training of model - yields predictions
    #     print('Beginning model training...')
    #     self.model.fit(**kwargs)
    #
    #     # set organization iterations counter
    #     org_ints = 1
    #
    #     # run update logic until passes criterion checks
    #     while not self.error_criterion() and not self.if_part_criterion():
    #         print('Organization iteration {}...'.format(org_ints))
    #
    #         # run criterion checks and organize accordingly
    #         self.organize(**kwargs)
    #
    #         # quit if above max neurons allowed
    #         if self.model.neurons >= self.max_neurons:
    #             print('Maximum neurons reached')
    #             print('Terminating self-organizing process')
    #
    #         # increase counter
    #         org_ints += 1
    #
    #     # print terminal message if successfully organized
    #     print('Self-Organization complete!')
    #     print('If-Part Criterion and Error Criterion both satisfied')

    def organize(self, x, **kwargs) -> None:
        """
        Run one iteration of organizational logic
        - check on system error and if-part criterion
        - add neurons or prune if needed
        """
        # create simple alias for self.model
        fuzzy_net = self.model

        # get copy of initial fuzzy weights
        start_weights = fuzzy_net.fuzz.get_weights()

        # widen centers if necessary
        if not self.if_part_criterion(x):
            self.widen_centers(x)

        # # add neuron if necessary
        # if not fuzzy_net.error_criterion():
        #     # reset fuzzy weights if previously widened before adding
        #     curr_weights = fuzzy_net.get_layer_weights(1)
        #     if not np.array_equal(start_weights, curr_weights):
        #         fuzzy_net.get_layer(1).set_weights(start_weights)
        #
        #     # add neuron and retrain model (if added)
        #     added = self.add_neuron(**kwargs)
        #     if added:
        #         self.train_model(**kwargs)
        #
        # # prune neurons and retrain model (if pruned)
        # pruned = self.prune_neurons(**kwargs)
        # if pruned:
        #     self.train_model(**kwargs)
        #
        # # check if needing to combine membership functions
        # self.combine_membership_functions(**kwargs)
        #

    def error_criterion(self, y_true, y_pred) -> bool:
        """
        Check error criterion for neuron-adding process.
            - considers generalization performance of model

        Returns
        =======
        - True:
            if criteron satisfied and no need to grow neuron
        - False:
            if criteron not met and need to add neuron
        """
        # mean of absolute test difference
        #y_pred = self.model_predictions()
        return mean_absolute_error(y_true, y_pred) <= self.err_delta

    def if_part_criterion(self, x) -> bool:
        """
        Check if-part criterion for neuron-adding process.
            - considers whether current fuzzy rules suitably cover inputs

            - get max of all neuron outputs (pre-normalization)
            - test whether max val at or above threshold
            - overall criterion met if criterion met for "ifpart_samples" % of samples

        Returns
        =======
        - True:
            if criteron satisfied and no need to widen centers
        - False:
            if criteron not met and need to widen neuron centers
        """
        # get max val
        fuzz_out = self.model.get_layer('FuzzyRules')(x)
        # check if max neuron output is above threshold
        maxes = np.max(fuzz_out, axis=-1) >= self.ifpart_thresh
        # return True if at least half of samples agree
        return (maxes.sum() / len(maxes)) >= self.ifpart_samples

    def minimum_distance_vector(self, x) -> np.ndarray:
        """
        Get minimum distance vector

        Returns
        =======
        min_dist : np.array
            - average minimum distance vector across samples
            - shape: (features, neurons)
        """

        # get input values and fuzzy weights
        samples = x.shape[0]
        c, s = self.model.get_layer('FuzzyRules').get_weights()

        # align x and c and assert matching dims
        aligned_x = x.repeat(self.model.get_layer('FuzzyRules').neurons). \
            reshape(x.shape + (self.model.get_layer('FuzzyRules').neurons,))
        aligned_c = c.repeat(samples).reshape((samples,) + c.shape)

        # average the minimum distance across samples
        return np.abs(aligned_x - aligned_c).mean(axis=0)

    def duplicate_model(self) -> Model:
        """
        Create duplicate model as FuzzyNetwork with identical weights
        """
        # create duplicate model and update weights
        dupe_mod = clone_model(self.model)
        dupe_mod.set_weights(self.model.get_weights())
        return dupe_mod

    def widen_centers(self, x) -> bool:
        """
        Widen center of neurons to better cover data
        """
        # print alert of successful widening
        print('Widening centers...')

        # create simple alias for self.network
        fuzzy_net = self.model

        # get fuzzy layer and output to find max neuron output
        fuzz_layer = fuzzy_net.get_layer('FuzzyRules')

        # get old weights and create current weight vars
        c, s = fuzz_layer.get_weights()

        # repeat until if-part criterion satisfied
        # only perform for max iterations
        counter = 0
        while not self.if_part_criterion(x):

            counter += 1
            # check if max iterations exceeded prior to satisfying if-part-criterion
            if counter > self.max_widens:
                print(f'Max iterations reached: ({counter - 1})')
                break

            # get neuron with max-output for each sample
            # then select the most common one to update
            fuzz_out = fuzz_layer(x)
            maxes = np.argmax(fuzz_out, axis=-1)
            max_neuron = np.argmax(np.bincount(maxes.flat))

            # select minimum width to expand
            # and multiply by factor
            mf_min = s[:, max_neuron].argmin()
            s[mf_min, max_neuron] = self.ksig * s[mf_min, max_neuron]

            # update weights
            new_weights = [c, s]
            fuzz_layer.set_weights(new_weights)

        # print alert of successful widening
        if counter == 0:
            print('Centers not widened')
        else:
            print(f'Centers widened after {counter} iterations')

        return self.if_part_criterion(x)

    # def rebuild_model(self,
    #                   new_weights: np.ndarray,
    #                   new_neurons: int,
    #                   **kwargs) -> Model:
    #     """
    #     Create updated FuzzyNetwork by adding or pruning neurons and updating to new weights
    #     """
    #     # get config from current model and update output_dim of neuron layers
    #     config = self.model.get_config()
    #     for layer in config['layers']:
    #         if 'output_dim' in layer['config']:
    #             layer['config']['output_dim'] = new_neurons
    #
    #     # load new model from custom config data and load new weights
    #     custom_objects = {'FuzzyLayer': FuzzyLayer,
    #                       'NormalizeLayer': NormalizeLayer,
    #                       'WeightedLayer': WeightedLayer,
    #                       'OutputLayer': OutputLayer}
    #     new_model = Model.from_config(config, custom_objects=custom_objects)
    #     new_model.set_weights(new_weights)
    #
    #     # recompile model based on current model parameters
    #     optimizer = kwargs.get('optimizer', self.model.optimizer)
    #     loss = kwargs.get('loss', self.model.loss)
    #     metrics = kwargs.get('metrics', self.model.metrics)
    #     new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
    #
    #     # update neuron attribute
    #     self.network.neurons = new_neurons
    #     return new_model
    #
    # def add_neuron(self, **kwargs) -> bool:
    #     """
    #     Add one additional neuron to the network
    #         - new FuzzyLayer  weights will be added using minimum distance vector calculation
    #         - new WeightedLayer weights are always a new column of 1
    #     """
    #     if self.__debug:
    #         print('Adding neuron...')
    #
    #     # get current weights
    #     w = self.model.get_weights()
    #     c_curr, s_curr, a_curr = w[0], w[1], w[2]
    #
    #     # get weights for new neuron
    #     ck, sk = self.new_neuron_weights()
    #     # expand dim for stacking
    #     ck = np.expand_dims(ck, axis=-1)
    #     sk = np.expand_dims(sk, axis=-1)
    #     c_new = np.hstack((c_curr, ck))
    #     s_new = np.hstack((s_curr, sk))
    #
    #     # update a vector to include column of 1s
    #     a_add = np.ones((a_curr.shape[-2]))
    #     a_new = np.column_stack((a_curr, a_add))
    #
    #     # update weights to include new neurons
    #     w[0], w[1], w[2] = c_new, s_new, a_new
    #
    #     # update model and neurons
    #     self.network.model = self.rebuild_model(new_weights=w, new_neurons=self.network.neurons + 1, **kwargs)
    #     self.model = self.network.model
    #
    #     if self.__debug:
    #         print('Neuron successfully added! - {} current neurons...'.format(self.network.neurons))
    #     return True
    #
    # def prune_neurons(self, **kwargs) -> bool:
    #     """
    #     Prune any unimportant neurons per effect on RMSE
    #     """
    #     if self.__debug:
    #         print('Pruning neurons...')
    #
    #     # create simple alias for self.network
    #     fuzzy_net = self.network
    #
    #     # quit if only 1 neuron exists
    #     if fuzzy_net.neurons == 1:
    #         if self.__debug:
    #             print('Skipping pruning steps - only 1 neuron exists')
    #         return False
    #
    #     # get current training predictions
    #     preds = self.model.predict(fuzzy_net.X_train)
    #
    #     # calculate mean-absolute-error on training data
    #     E_rmse = mean_squared_error(fuzzy_net.y_train, preds)
    #
    #     # create duplicate model and get both sets of model weights
    #     prune_model = self.duplicate_model()
    #     act_weights = self.model.get_weights()
    #
    #     # for each neuron, zero it out in prune model
    #     # and get change in mae for dropping neuron
    #     delta_E = []
    #     for neuron in range(fuzzy_net.neurons):
    #         # reset prune model weights to actual weights
    #         prune_model.set_weights(act_weights)
    #
    #         # get current prune weights
    #         w = prune_model.get_weights()
    #         # zero our i neuron column in weighted vector
    #         a = w[2]
    #         a[:, neuron] = 0
    #         prune_model.set_weights(w)
    #
    #         # predict values with new zeroed out weights
    #         neur_pred = prune_model.predict(fuzzy_net.X_train)
    #         neur_rmae = mean_absolute_error(fuzzy_net.y_train, neur_pred)
    #
    #         # append difference in rmse and new prediction rmse
    #         delta_E.append(neur_rmae - E_rmse)
    #
    #     # convert delta_E to numpy array
    #     delta_E = np.array(delta_E)
    #     # choose max of tolerance or threshold limit
    #     E = max(self._prune_tol * E_rmse, self._k_rmse)
    #
    #     # iterate over each neuron in ascending importance
    #     # and prune until hit "important" neuron
    #     deleted = []
    #     # for each neuron up to second most important
    #     for neuron in delta_E.argsort()[:-1]:
    #         # reset prune model weights to actual weights
    #         prune_model.set_weights(act_weights)
    #
    #         # get current prune weights
    #         w = prune_model.get_weights()
    #         a = w[2]
    #         # zero out previous deleted neurons
    #         for delete in deleted:
    #             a[:, delete] = 0
    #         prune_model.set_weights(w)
    #
    #         # predict values with new zeroed out weights
    #         neur_pred = prune_model.predict(fuzzy_net.X_train)
    #         E_rmae_del = mean_absolute_error(fuzzy_net.y_train, neur_pred)
    #
    #         # if E_mae_del < E
    #         # delete neuron
    #         if E_rmae_del < E:
    #             deleted.append(neuron)
    #             continue
    #         # quit deleting if >= E
    #         else:
    #             break
    #
    #     # exit if no neurons to be deleted
    #     if not deleted:
    #         if self.__debug:
    #             print('No neurons detected for pruning')
    #         return False
    #     else:
    #         if self.__debug:
    #             print('Neurons to be deleted: ')
    #             print(deleted)
    #
    #     # reset prune model weights to actual weights
    #     prune_model.set_weights(act_weights)
    #     # get current prune weights and remove deleted neurons
    #     w = prune_model.get_weights()
    #     for i, weight in enumerate(w[:3]):
    #         w[i] = np.delete(weight, deleted, axis=-1)
    #
    #     # update model with updated weights
    #     self.network.model = self.rebuild_model(new_weights=w, new_neurons=self.network.neurons - len(deleted),
    #                                             **kwargs)
    #     self.model = self.network.model
    #
    #     if self.__debug:
    #         print('{} neurons successfully pruned! - {} current neurons...'.
    #               format(len(deleted), self.network.neurons))
    #     return True
    #
    # def new_neuron_weights(self, dist_thresh: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Return new c and s weights for k new fuzzy neuron
    #
    #     Parameters
    #     ==========
    #     dist_thresh : float
    #         - multiplier of average features values to use as distance thresholds
    #
    #     Returns
    #     =======
    #     ck : np.array
    #         - average minimum distance vector across samples
    #         - shape: (features,)
    #     sk : np.array
    #         - average minimum distance vector across samples
    #         - shape: (features,)
    #     """
    #
    #     # create simple alias for self.network
    #     fuzzy_net = self.network
    #
    #     # get input values and fuzzy weights
    #     x = fuzzy_net.X_train
    #     c, s = fuzzy_net.get_layer_weights(1)
    #
    #     # get minimum distance vector
    #     min_dist = self.min_dist_vector()
    #     # get minimum distance across neurons
    #     # and arg-min for neuron with lowest distance
    #     dist_vec = min_dist.min(axis=-1)
    #     min_neurs = min_dist.argmin(axis=-1)
    #
    #     # get min c and s weights
    #     c_min = c[:, min_neurs].diagonal()
    #     s_min = s[:, min_neurs].diagonal()
    #     assert c_min.shape == s_min.shape
    #
    #     # set threshold distance as factor of mean
    #     # value for each feature across samples
    #     kd_i = x.mean(axis=0) * dist_thresh
    #
    #     # get final weight vectors
    #     ck = np.where(dist_vec <= kd_i, c_min, x.mean(axis=0))
    #     sk = np.where(dist_vec <= kd_i, s_min, dist_vec)
    #     return ck, sk
    #
    # # TODO: add method combining membership functions
    # def combine_membership_functions(self, **kwargs) -> None:
    #     """
    #     Function to combine redundant membership functions to simplify training parameters
    #     """
    #     pass
