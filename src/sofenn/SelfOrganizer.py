import inspect
import logging
from typing import Tuple, Optional

import numpy
from keras.api.metrics import CategoricalAccuracy, MeanSquaredError, Accuracy
from keras.api.models import clone_model
from keras.api.optimizers import Adam, RMSprop
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sofenn.FuzzyNetwork import FuzzyNetwork
from sofenn.losses import CustomLoss

logger = logging.getLogger(__name__)

class FuzzySelfOrganizer(object):
    """
    Self-Organizing Fuzzy Neural Network
    ====================================
    Meta-model for fuzzy neural network that updates model architecture in addition to parameter tuning.

    Implemented per description in:

    "An on-line algorithm for creating self-organizing fuzzy neural networks" - Leng, Prasad, McGinnity (2004).
    Composed of 5 layers with varying "fuzzy rule" nodes.

    :param max_neurons: Maximum neurons allowed during organizing.
    :param ifpart_threshold: If-Part threshold.
    :param ifpart_samples: If-Part samples.
    :param error_delta: Error delta.
    :param k_sigma: Widening factor during center-widening.
    :param max_widens: Maximum number of center-widening loops.
    :param prune_tol: Pruning tolerance limit of total root mean squared error.
    :param k_rmse: Expected root mean squared error when pruning neurons.

    Methods
    =======
    self_organize:
        Run self-organizing logic to determine optimal network structure.
    organize:
        One iteration of self-organizing logic.
    error_criterion:
        Check the error criterion for the neuron-adding process.
        Criterion considers generalization performance of the model.
    if_part_criterion:
        Check the if-part criterion for the neuron-adding process.
        Criterion considers whether current fuzzy rules suitably cover inputs.
    minimum_distance_vector:
        Calculate the minimum distance vector between inputs and current neuron centers.
    duplicate_model:
        Duplicate the fuzzy neural network model with identical weights.
    widen_centers:
        Widen the neuron centers to better cover input data.
    add_neuron:
        Add one neuron to the network.
    new_neuron_weights:
        Return new centers and widths for new fuzzy neuron.
    rebuild_model:
        Create updated fuzzy network when adding or pruning neurons, and update weights.
    prune_neurons:
        Prune any unimportant neurons as measured by their contribution to root mean squared error.
    """

    def __init__(
            self,
            model: Optional[FuzzyNetwork] = None,
            max_loops: int = 10,
            max_neurons: int = 100,
            ifpart_threshold: float = 0.1354,
            ifpart_samples: float = 0.95,
            error_delta: float = 0.12,
            k_sigma: float = 1.12,
            max_widens: float = 250,
            prune_tol: float = 0.8,
            k_rmse:float = 0.1,
            **kwargs
    ):
        if max_loops < 0:
            raise ValueError(f"Maximum organizing loops cannot be less than 0.")
        self.max_loops = max_loops

        initial_neurons = kwargs.get('neurons', inspect.signature(FuzzyNetwork).parameters['neurons'].default)
        if max_neurons < initial_neurons:
            raise ValueError(f"Maximum neurons cannot be smaller than initialized neurons: {initial_neurons}.")
        self.max_neurons = max_neurons

        if ifpart_threshold < 0:
            raise ValueError(f"If-Part Threshold must not be negative: {ifpart_threshold}")
        self.ifpart_threshold = ifpart_threshold

        if not 0 < ifpart_samples <= 1.0:
            raise ValueError(f'If-Part Samples must be between between 0 and 1: {ifpart_samples}')
        self.ifpart_samples = ifpart_samples

        if error_delta < 0:
            raise ValueError(f"Error Delta must not be negative: {error_delta}")
        self.error_delta = error_delta

        if k_sigma <= 1:
            raise ValueError('Widening factor (k_sigma) must be larger than 1')
        if max_widens < 0:
            raise ValueError('Max center widens must be at least 0')
        if not 0 < prune_tol < 1:
            raise ValueError('Prune tolerance must be between 0 and 1')
        if k_rmse <= 0:
            raise ValueError('Expected RMSE must be greater than 0')
        self.k_sigma = k_sigma
        self.max_widens = max_widens
        self.prune_tol = prune_tol
        self.k_rmse = k_rmse

        self.model = FuzzyNetwork(**kwargs) if model is None else model

    def self_organize(self, x, y, **kwargs) -> bool:
        """
        Run self-organizing logic to determine optimal network structure.

        Train initial model in parameters then begin self-organization.

        * If the If-Part criterion fails, widen rule widths.
        * If the If-Part criterion still fails, reset to original widths, then add neuron and retrain weights.

        :param: x: Input Data.
        :param: y: Target Data.

        :return: True if the resulting model after organizing satisfies both error and if-part criteria.
        """
        # initial training of model - yields predictions
        print('Beginning model training...')
        self.model.fit(x, y, **kwargs)

        # set organization iterations counter
        org_ints = 1

        # run update logic until passes criterion checks
        while not (self.error_criterion(y, self.model.predict(x)) and self.if_part_criterion(x)):
            if org_ints > self.max_loops:
                break

            print('Organization iteration {}...'.format(org_ints))

            # run criterion checks and organize accordingly
            self.organize(x, y, **kwargs)

            # quit if above max neurons allowed
            if self.model.neurons >= self.max_neurons:
                print('Maximum neurons reached')
                print('Terminating self-organizing process')
                break

            # increase counter
            org_ints += 1

        # print terminal message if successfully organized
        print('Self-Organization complete!')
        return self.error_criterion(y, self.model.predict(x)) and self.if_part_criterion(x)

    def organize(self, x, y, **kwargs) -> None:
        """
        One iteration of self-organizing logic.

        * Check system error and if-part criterion.
        * Add neurons or prune if needed.

        :param: x: Input data.
        :param: y: Target data.
        """
        if 'epochs' not in kwargs:
            kwargs['epochs'] = 10

        # no structural adjustment
        if self.error_criterion(y, self.model.predict(x)) and self.if_part_criterion(x):
            self.model.fit(x, y, **kwargs)
        # widen MF widths to cover input vector of MF with lowest value
        elif self.error_criterion(y, self.model.predict(x)) and not self.if_part_criterion(x):
            self.widen_centers(x)
        # add neuron and retrain after adding neuron
        elif not self.error_criterion(y, self.model.predict(x)) and self.if_part_criterion(x):
            self.add_neuron(x, y, **kwargs)
            self.model.fit(x, y, **kwargs)
        # widen centers until if-part satisfied. if if-else not satisfied, reset widths and add neuron
        elif not self.error_criterion(y, self.model.predict(x)) and not self.if_part_criterion(x):
            original_weights = self.model.get_weights()
            self.widen_centers(x)
            if not self.if_part_criterion(x):
                self.model.set_weights(original_weights)
                self.add_neuron(x, y, **kwargs)
                self.model.fit(x, y, **kwargs)

        # prune neurons and retrain model (if pruned)
        pruned = self.prune_neurons(x, y, **kwargs)
        if pruned:
            self.model.fit(x, y, **kwargs)

        # combine membership functions where appropriate
        #self.combine_membership_functions(**kwargs)

    def error_criterion(self, y_true, y_pred) -> bool:
        """
        Check the error criterion for the neuron-adding process.
        Criterion considers generalization performance of the model.

        :param y_true: True values for target dataset.
        :param y_pred: Predicted target dataset.

        :returns: True/False if the error criterion is satisfied.
            If the criterion is not satisfied, then a neuron should be added to the model.
        """
        return mean_absolute_error(y_true, y_pred) <= self.error_delta

    def if_part_criterion(self, x) -> bool:
        """
        Check the if-part criterion for the neuron-adding process.
        Criterion considers whether current fuzzy rules suitably cover inputs.

        :param x: Input data.

        :returns: True/False if the if-part criterion is satisfied.
            If the criterion is not satisfied, then the existing neuron centers should be widened.
        """
        # get fuzzy output
        fuzz_out = self.model.fuzz(x)
        # check if max neuron output is above the if-part threshold
        maxes = numpy.max(fuzz_out, axis=-1) >= self.ifpart_threshold
        # return True if the proportion of samples above the if-part threshold exceeds the required sample proportion
        return (maxes.sum() / len(maxes)) >= self.ifpart_samples

    def minimum_distance_vector(self, x) -> numpy.ndarray:
        """
        Calculate the minimum distance vector between inputs and current neuron centers.

        :param x: Input data.

        :returns: Minimum Distance Vector: Average minimum distance vector across samples. Shape: (features, neurons).
        """

        # get input values and fuzzy weights
        samples = x.shape[0]
        c, s = self.model.fuzz.get_weights()

        # align x and c and assert matching dims
        aligned_x = x.repeat(self.model.neurons).reshape(x.shape + (self.model.neurons,))
        aligned_c = c.repeat(samples).reshape((samples,) + c.shape)

        # average the minimum distance across samples
        return numpy.abs(aligned_x - aligned_c).mean(axis=0)

    def duplicate_model(self) -> FuzzyNetwork:
        """
        Duplicate the fuzzy neural network model with identical weights.

        :returns: Duplicated fuzzy neural network model.
        """
        # create duplicate model and update weights
        dupe_mod = clone_model(self.model)
        dupe_mod.set_weights(self.model.get_weights())
        return dupe_mod

    def widen_centers(self, x) -> bool:
        """
        Widen the neuron centers to better cover input data.

        :param x: Input data.

        :returns: True if centers successfully widened. False otherwise.
        """
        # print alert of successful widening
        print('Widening centers...')

        # create simple alias for self.network
        fuzzy_net = self.model

        # get fuzzy layer and output to find max neuron output
        fuzz_layer = fuzzy_net.fuzz

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
            maxes = numpy.argmax(fuzz_out, axis=-1)
            max_neuron = numpy.argmax(numpy.bincount(maxes.flat))

            # select minimum width to expand
            # and multiply by factor
            mf_min = s[:, max_neuron].argmin()
            s[mf_min, max_neuron] = self.k_sigma * s[mf_min, max_neuron]

            # update weights
            new_weights = [c, s]
            fuzz_layer.set_weights(new_weights)

        # print alert of successful widening
        if counter == 0:
            print('Centers not widened')
        else:
            print(f'Centers widened after {counter} iterations')

        return self.if_part_criterion(x)

    def add_neuron(self, x, y, **kwargs) -> bool:
        """
        Add one neuron to the network.

        - New FuzzyLayer weights will be added using minimum distance vector calculation.
        - New WeightedLayer weights are always a new column of 1.

        :param x: Input data.
        :param y: Output data.

        :returns: True if a neuron was added. False otherwise.
        """
        # if self.__debug:
        #     print('Adding neuron...')
        #pass

        # get current weights
        w = self.model.get_weights()
        c_curr, s_curr, a_curr = w[0], w[1], w[2]

        # get weights for new neuron
        ck, sk = self.new_neuron_weights(x)
        # expand dim for stacking
        ck = numpy.expand_dims(ck, axis=-1)
        sk = numpy.expand_dims(sk, axis=-1)
        c_new = numpy.hstack((c_curr, ck))
        s_new = numpy.hstack((s_curr, sk))

        # TODO: confirm new logic of adding weights A for new neuron.
        #       currently taking average of existing weights, but explore other options
        #       1 - (current) use average of existing weights
        #       2 - initialize A weights as 0 or 1
        # update a vector to include column of 1s
        #a_add = np.ones((a_curr.shape[0]))
        #a_new = np.column_stack((a_curr, a_add))
        a_add = a_curr.mean(axis=0)
        a_new = numpy.row_stack((a_curr, a_add))

        # update weights to include new neurons
        w[0], w[1], w[2] = c_new, s_new, a_new

        # update model and neurons
        self.model = self.rebuild_model(x=x, y=y, new_neurons=self.model.neurons + 1, new_weights=w, **kwargs)
        #self.model = self.network.model

        #if self.__debug:
        print('Neuron successfully added! - {} current neurons...'.format(self.model.neurons))
        return True

    def new_neuron_weights(self, x, distance_threshold: float = 1.0) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Return new centers and widths for new fuzzy neuron.

        :param x: Input data.
        :param distance_threshold: Multiplier of average features values to use as distance thresholds.

        :returns: (ck, sk): Centers and widths for new fuzzy neuron. Shape: (features,)
        """
        # get input values and fuzzy weights
        c, s = self.model.get_layer('FuzzyRules').get_weights()

        # get minimum distance vector
        minimum_distance = self.minimum_distance_vector(x)
        # get minimum distance across neurons and arg-min for neuron with the lowest distance
        distance_vector = minimum_distance.min(axis=-1)
        minimum_neurons = minimum_distance.argmin(axis=-1)

        # get min c and s weights
        c_min = c[:, minimum_neurons].diagonal()
        s_min = s[:, minimum_neurons].diagonal()

        # set threshold distance as a factor of mean
        # value for each feature across samples
        kd_i = x.mean(axis=0) * distance_threshold

        # get final weight vectors
        ck = numpy.where(distance_vector <= kd_i, c_min, x.mean(axis=0))
        sk = numpy.where(distance_vector <= kd_i, s_min, distance_vector)
        return ck, sk

    def rebuild_model(self,x, y, new_neurons: int, new_weights: list[numpy.ndarray], **kwargs) -> FuzzyNetwork:
        """
        Create updated fuzzy network when adding or pruning neurons, and update weights.

        :param: x: Input data.
        :param: y: Output data.
        :param: new_neurons: Number of neurons in the new rebuilt model.
        :param: new_weights: New weights for the rebuilt model.

        :returns: Rebuilt fuzzy network according to new specifications.
        """
        # get config from current model and update output_dim of neuron layers
        config = self.model.get_config()
        config['neurons'] = new_neurons
        new_model = FuzzyNetwork(**config)

        # recompile model based on current model parameters
        if self.model.problem_type == 'classification':
            # TODO: create mapping dictionary of problem type and default loss/optimizer/metrics
            default_loss = CustomLoss()
            default_optimizer = Adam()
            default_metrics = [CategoricalAccuracy()]
            # if self.y_test.ndim == 2:                       # binary classification
            #     default_metrics = ['binary_accuracy']
            # else:                                           # multi-class classification
            #     default_metrics = ['categorical_accuracy']
        else:
            default_loss = MeanSquaredError()
            default_optimizer = RMSprop()
            default_metrics = [Accuracy()]
        loss = kwargs.pop('loss', default_loss)
        optimizer = kwargs.pop('optimizer', default_optimizer)
        metrics = kwargs.pop('metrics', default_metrics)

        compile_args = list(inspect.signature(FuzzyNetwork.compile).parameters)
        compile_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in compile_args}

        fit_args = list(inspect.signature(FuzzyNetwork.fit).parameters)
        fit_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in fit_args}

        new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **compile_dict)
        new_model.fit(x, y, epochs=1, **fit_dict)
        new_model.set_weights(new_weights)

        return new_model

    def prune_neurons(self, x, y, **kwargs) -> bool:
        """
        Prune any unimportant neurons as measured by their contribution to root mean squared error.

        :param x: Input data.
        :param y: Output data.

        :returns: True if at least one neuron was pruned. False otherwise.
        """
        print('Pruning neurons...')

        # quit if only 1 neuron exists
        if self.model.neurons == 1:
            print('Skipping pruning steps - only 1 neuron exists')
            return False

        # get current training predictions
        # calculate mean absolute error on training data
        E_rmse = mean_squared_error(y, self.model.predict(x))

        # create a duplicate model and get both sets of model weights
        prune_model = self.duplicate_model()
        starting_weights = self.model.get_weights()

        # for each neuron, zero it out in the prune model
        # and get change in mae for dropping neuron
        delta_E = []
        for neuron in range(self.model.neurons):
            # reset prune model weights to actual weights
            prune_model.set_weights(starting_weights)

            # get current prune weights
            w = prune_model.get_weights()
            # zero our i neuron column in weighted vector
            a = w[2]
            a[neuron, :] = 0
            prune_model.set_weights(w)

            # predict values with new zeroed out weights
            neuron_rmae = mean_absolute_error(y, prune_model.predict(x))

            # append difference in rmse and new prediction rmse
            delta_E.append(neuron_rmae - E_rmse)

        # convert delta_E to numpy array
        delta_E = numpy.array(delta_E)
        # choose max of tolerance or threshold limit
        E = max(self.prune_tol * E_rmse, self.k_rmse)

        # iterate over each neuron in ascending importance
        # and prune until hit "important" neuron
        to_delete = []
        # for each neuron excluding most important
        for neuron in delta_E.argsort()[:-1]:
            # reset prune model weights to actual weights
            prune_model.set_weights(starting_weights)

            # get current prune weights
            w = prune_model.get_weights()
            a = w[2]
            # zero out previous deleted neurons
            a[neuron, :] = 0
            prune_model.set_weights(w)

            # predict values with new zeroed out weights
            E_rmae_del = mean_absolute_error(y, prune_model.predict(x))

            # if E_mae_del < E
            # delete neuron
            if E_rmae_del < E:
                to_delete.append(neuron)
                continue
            # quit deleting if >= E
            else:
                break
        # exit if no neurons to be deleted
        if not to_delete:
            print('No neurons detected for pruning')
            return False
        else:
            print(f'Neurons to be deleted: {to_delete}')

        # reset prune model weights to actual weights
        prune_model.set_weights(starting_weights)
        # get current prune weights and remove deleted neurons
        w = prune_model.get_weights()
        for i, weight in enumerate(w[:2]):
            w[i] = numpy.delete(weight, to_delete, axis=-1)
        w[2] = numpy.delete(w[2], to_delete, axis=0)

        # update model with updated weights
        self.model = self.rebuild_model(x, y, new_neurons=self.model.neurons - len(to_delete), new_weights=w,**kwargs)

        print(f'{len(to_delete)} neurons successfully pruned! - {self.model.neurons} current neurons...')
        return True

    def combine_membership_functions(self, **kwargs) -> None:
        """
        Combine redundant membership functions to simplify training parameters
        """
        raise NotImplementedError
