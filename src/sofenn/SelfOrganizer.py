import inspect
import logging
from typing import Tuple, Optional

import numpy
from keras.metrics import MeanSquaredError, MeanAbsoluteError
from keras.models import Model
from keras.models import clone_model

from sofenn.FuzzyNetwork import FuzzyNetwork
from sofenn.utils.layers import parse_function_kwargs

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
            raise ValueError(f"Maximum organizing loops cannot be less than 0")
        self.max_loops = max_loops

        initial_neurons = kwargs.get('neurons', inspect.signature(FuzzyNetwork).parameters['neurons'].default) \
            if model is None else model.neurons
        if max_neurons < initial_neurons:
            raise ValueError("Maximum neurons cannot be smaller than initialized neurons: {max_neurons} < {initial_neurons}")
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

        self.weight_index = {
            'centers': 0,
            'widths':  1,
            'weights': 2
        }

        self.model = model if model is not None else FuzzyNetwork(**kwargs)

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
        logger.info('Beginning self-organizing process')

        fit_kwargs = parse_function_kwargs(kwargs, Model.fit)

        self.model.fit(x, y, **fit_kwargs)

        # set organization iterations counter
        organization_iterations = 1

        # run update logic until passes criterion checks
        while not (self.error_criterion(y, self.model.predict(x)) and self.if_part_criterion(x)):
            if organization_iterations > self.max_loops:
                break
            logger.debug(f'Running self-organize iteration: {organization_iterations}')
            self.organize(x, y, **kwargs)
            if self.model.neurons >= self.max_neurons:
                logger.info(f'Maximum neurons reached: {self.max_neurons}. Terminating self-organizing process')
                break
            organization_iterations += 1

        logger.info('Self-organization complete!')
        logger.debug(f'Organization completed after {organization_iterations} iterations')
        return self.error_criterion(y, self.model.predict(x)) and self.if_part_criterion(x)

    def organize(self, x, y, **kwargs) -> None:
        """
        One iteration of self-organizing logic.

        * Check system error and if-part criterion.
        * Add neurons or prune if needed.

        :param: x: Input data.
        :param: y: Target data.
        """
        fit_kwargs = parse_function_kwargs(kwargs, Model.fit)

        error_criterion = self.error_criterion(y, self.model.predict(x))
        ifpart_criterion = self.if_part_criterion(x)
        logger.debug(f'Error criterion satisfied: {error_criterion}')
        logger.debug(f'If-Part criterion satisfied: {ifpart_criterion}')

        # no structural adjustment
        if error_criterion and ifpart_criterion:
            self.model.fit(x, y, **fit_kwargs)
        # widen membership function widths to cover the input vector of membership function with the lowest value
        elif error_criterion and not ifpart_criterion:
            self.widen_centers(x)
        # add neuron and retrain after adding neuron
        elif not error_criterion and ifpart_criterion:
            self.add_neuron(x, y, **kwargs)
            self.model.fit(x, y, **fit_kwargs)
        # widen centers until if-part satisfied. if if-part not satisfied, reset widths and add neuron
        elif not error_criterion and not ifpart_criterion:
            original_weights = self.model.get_weights()
            self.widen_centers(x)
            if not self.if_part_criterion(x):
                self.model.set_weights(original_weights)
                self.add_neuron(x, y, **kwargs)
                self.model.fit(x, y, **fit_kwargs)

        # prune neurons and retrain model (if pruned)
        pruned = self.prune_neurons(x, y, **kwargs)
        if pruned:
            self.model.fit(x, y, **fit_kwargs)

        # TODO: add combining of membership functions
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
        return MeanAbsoluteError()(y_true, y_pred).numpy() <= self.error_delta

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

    def widen_centers(self, x) -> bool:
        """
        Widen the neuron centers to better cover input data.

        :param x: Input data.

        :returns: True if centers successfully widened. False otherwise.
        """
        logger.info('Beginning process to widen centers')

        c, s = self.model.fuzz.get_weights()

        counter = 0
        while not self.if_part_criterion(x):
            counter += 1
            if counter > self.max_widens:
                logger.info(f'Max iterations reached: ({counter - 1})')
                break

            # get neuron with max-output for each sample then select the most common one to update
            fuzz_out = self.model.fuzz(x)
            maxes = numpy.argmax(fuzz_out, axis=-1)
            max_neuron = numpy.argmax(numpy.bincount(maxes.flat))

            # select minimum width to expand and multiply by factor
            mf_min = s[:, max_neuron].argmin()
            s[mf_min, max_neuron] = self.k_sigma * s[mf_min, max_neuron]

            new_weights = [c, s]
            self.model.fuzz.set_weights(new_weights)

        if counter == 0:
            logger.info('Centers not widened because if-part criterion already satisfied')
        else:
            logger.info(f'Centers successfully widened after {counter} iterations')

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
        logger.info('Adding neuron')

        w = self.model.get_weights()
        c_curr = w[self.weight_index['centers']]
        s_curr = w[self.weight_index['widths']]
        a_curr = w[self.weight_index['weights']]

        # get weights for new neuron
        ck, sk = self.new_neuron_weights(x)
        # expand dim for stacking
        ck = numpy.expand_dims(ck, axis=-1)
        sk = numpy.expand_dims(sk, axis=-1)
        c_new = numpy.hstack((c_curr, ck))
        s_new = numpy.hstack((s_curr, sk))

        a_add = a_curr.mean(axis=0)
        a_new = numpy.vstack((a_curr, a_add))

        w[self.weight_index['centers']] = c_new
        w[self.weight_index['widths']] = s_new
        w[self.weight_index['weights']] = a_new

        self.model = self.rebuild_model(x, y, new_neurons=self.model.neurons + 1, new_weights=w, **kwargs)

        logger.info(f'Neuron successfully added!')
        logger.debug(f'Current neurons: {self.model.neurons}')
        return True

    def new_neuron_weights(self, x, distance_threshold: float = 1.0) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Return new centers and widths for new fuzzy neuron.

        :param x: Input data.
        :param distance_threshold: Multiplier of average features values to use as distance thresholds.

        :returns: (ck, sk): Centers and widths for new fuzzy neuron. Shape: (features,)
        """
        c, s = self.model.fuzz.get_weights()

        minimum_distance = self.minimum_distance_vector(x)
        distance_vector = minimum_distance.min(axis=-1)
        minimum_neurons = minimum_distance.argmin(axis=-1)

        c_min = c[:, minimum_neurons].diagonal()
        s_min = s[:, minimum_neurons].diagonal()

        # set threshold distance as a factor of mean value for each feature across samples
        kd_i = x.mean(axis=0) * distance_threshold

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
        new_model_config = self.model.get_config()
        new_model_config['neurons'] = new_neurons
        logger.debug(f'Rebuilding model with config: {new_model_config}')
        new_model = FuzzyNetwork.from_config(new_model_config)

        compile_config = self.model.get_compile_config()
        logger.debug(f'Rebuilding model with config: {compile_config}')
        new_model.compile_from_config(compile_config)

        # run training for 1 interval so that model weights are re-initialized to accommodate new neuron
        fit_kwargs = parse_function_kwargs(kwargs, Model.fit)
        if 'epochs' in fit_kwargs:
            logger.warning(f'Ignoring provided value for epochs: {fit_kwargs["epochs"]}. '
                           f'Will set epochs to 1 for rebuilding')
        fit_kwargs['epochs'] = 1
        new_model.fit(x, y, **fit_kwargs)

        new_model.set_weights(new_weights)
        return new_model

    def prune_neurons(self, x, y, **kwargs) -> bool:
        """
        Prune any unimportant neurons as measured by their contribution to root mean squared error.

        :param x: Input data.
        :param y: Output data.

        :returns: True if at least one neuron was pruned. False otherwise.
        """
        logger.info('Beginning neuron pruning')

        if self.model.neurons == 1:
            logger.info('Skipping pruning step - only 1 neuron currently exists')
            return False

        # calculate mean absolute error on training data
        E_rmse = MeanSquaredError()(y, self.model.predict(x))

        # create a duplicate model to use while measuring the effects of pruning
        starting_weights = self.model.get_weights()
        prune_model = clone_model(self.model)
        prune_model.set_weights(starting_weights)

        # for each neuron, zero it out in the prune model and get change in root mean squared error from removing neuron
        delta_E = []
        for neuron in range(self.model.neurons):
            # reset prune model weights to actual weights
            prune_model.set_weights(starting_weights)

            # get current prune weights
            w = prune_model.get_weights()
            # zero our i neuron column in weighted vector
            a = w[self.weight_index['weights']]
            a[neuron, :] = 0
            prune_model.set_weights(w)

            # predict values with new zeroed out weights
            neuron_rmae = MeanAbsoluteError()(y, prune_model.predict(x))

            # append difference in rmse and new prediction rmse
            delta_E.append(neuron_rmae - E_rmse)

        delta_E = numpy.array(delta_E)
        # choose max of pruning tolerance or threshold limit
        E = max(self.prune_tol * E_rmse, self.k_rmse)

        # iterate over each neuron in ascending importance and prune until hit important neuron
        to_delete = []
        # for each neuron excluding the most important
        for neuron in delta_E.argsort()[:-1]:
            # reset prune model weights to actual weights
            prune_model.set_weights(starting_weights)

            # get current prune weights
            w = prune_model.get_weights()
            a = w[self.weight_index['weights']]
            # zero out previous deleted neurons
            a[neuron, :] = 0
            prune_model.set_weights(w)

            # predict values with new zeroed out weights
            E_rmae_del = MeanAbsoluteError()(y, prune_model.predict(x))

            # if E_mae_del < E
            # delete neuron
            if E_rmae_del < E:
                to_delete.append(neuron)
                continue
            # quit deleting if >= E
            else:
                break

        if not to_delete:
            logger.info('No neurons detected for pruning')
            return False
        else:
            logger.info(f'Neurons to be deleted: {to_delete}')

        # reset prune model weights to actual weights
        prune_model.set_weights(starting_weights)
        # get current prune weights and remove deleted neurons
        w = prune_model.get_weights()
        for fuzzy_weight in ['centers', 'widths']:
            w[self.weight_index[fuzzy_weight]] = numpy.delete(w[self.weight_index[fuzzy_weight]], to_delete, axis=-1)
        w[self.weight_index['weights']] = numpy.delete(w[self.weight_index['weights']], to_delete, axis=0)

        self.model = self.rebuild_model(x, y, new_neurons=self.model.neurons - len(to_delete), new_weights=w, **kwargs)
        logger.info(f'{len(to_delete)} neurons successfully pruned')
        logger.debug(f'Current neurons: {self.model.neurons}')
        return True

    def combine_membership_functions(self, **kwargs) -> None:
        """
        Combine redundant membership functions to simplify training parameters
        """
        raise NotImplementedError
