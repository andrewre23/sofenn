import warnings
from typing import Optional, Union

import keras.api.ops as K
import numpy as np
from keras.api.layers import Input, Dense
from keras.api.models import Model
from keras.api.utils import to_categorical

from sofenn.layers import FuzzyLayer, NormalizeLayer, WeightedLayer, OutputLayer


def FuzzyNetwork(
        input_tensor,
        input_shape: tuple,
        neurons: int = 1,
        max_neurons: int = 100,
        problem_type: str = 'classification',
        target_classes: Optional[int] = 1,
        **kwargs
):
    # validate input tensor / shape
    if input_tensor and input_shape:
        raise ValueError("input_tensor and input_shape are mutually exclusive")
    features = input_tensor.shape[-1] if input_tensor else input_shape[-1]

    if neurons <= 0:
        raise ValueError(f"'neurons' must be positive integer. Passed: {neurons}.")
    neurons = neurons
    if max_neurons < neurons:
        raise ValueError(f"'max_neurons' must be positive integer no greater than 'neurons'({neurons}). "
                         f"Provided: {max_neurons}.")
    max_neurons = max_neurons

    if problem_type.lower() == 'classification':
        if not target_classes >= 1:
            raise ValueError(f"'target_classes' must be greater than 1. Provided: {target_classes}.")
    if problem_type.lower() == 'regression':
        if target_classes is not None:
            warnings.warn(f"Provided value for 'target_classes' ({target_classes}), "
                          f"but parameter is ignored when 'problem_type' is 'classification'.")
            target_classes = None
    else:
        raise ValueError("Invalid problem type.")
    problem_type = problem_type
    target_classes = target_classes


    model = Model(inputs=inputs, outputs=final_out, name='FuzzyNetwork', **kwargs)



# TODO: remove X/y train/test as inputs and replace with both input_tensor and input_shape as optional inputs
class FuzzyNetworkModel(Model):
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
    - ifpart_samples : float
        - percent of samples needed to meet ifpart criterion
    - err_delta : float
        - threshold for error criterion whether new neuron to be added
    - debug : boolean
        - debug flag

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
    def __init__(self,
#                 input_tensor: Optional[K.tensor],
                 input_shape: tuple,
                 neurons: int = 1,
                 max_neurons: int = 100,
                 problem_type: str = 'classification',
                 target_classes: Optional[int] = 1,
                 **kwargs):

        # validate input tensor / shape
        #if input_tensor and input_shape:
            #raise ValueError("input_tensor and input_shape are mutually exclusive")
        self.features = input_shape[-1]
        #self.features = input_tensor.shape[-1] if input_tensor else input_shape[-1]

        if neurons <= 0:
            raise ValueError(f"'neurons' must be positive integer. Passed: {neurons}.")
        self.neurons = neurons
        if max_neurons < neurons:
            raise ValueError(f"'max_neurons' must be positive integer no greater than 'neurons'({neurons}). "
                             f"Provided: {max_neurons}.")
        self.max_neurons = max_neurons

        if problem_type.lower() == 'classification':
            if not (target_classes >= 1):
                raise ValueError(f"'target_classes' must be greater than 1. Provided: {target_classes}.")
        elif problem_type.lower() == 'regression':
            if target_classes is not None:
                warnings.warn(f"Provided value for 'target_classes' ({target_classes}), "
                              f"but parameter is ignored when 'problem_type' is 'classification'.")
                target_classes = None
        else:
            raise ValueError("Invalid problem type.")
        self.problem_type = problem_type
        self.target_classes = target_classes

        super().__init__()
        # add layers
        self.inputs = Input(name='Inputs', shape=(self.features,))
        self.fuzz = FuzzyLayer(shape=(self.features,), neurons=self.neurons)
        self.norm = NormalizeLayer(shape=(self.features, self.neurons))
        self.w = WeightedLayer(shape=[(self.features,), (self.neurons,)])
        self.raw = OutputLayer()

    def call(self, inputs):
        phi = self.fuzz(inputs)
        psi = self.norm(phi)
        f = self.w([inputs, psi])
        raw_output = self.raw(f)
        final_out = raw_output

        # add softmax layer for classification problem
        if self.problem_type == 'classification':
            classify = Dense(self.target_classes, name='Softmax', activation='softmax')
            classes = classify(raw_output)
            final_out = classes
        return final_out

    def compile_model(self,
                      init_c: bool = True,
                      random: bool = True,
                      init_s: bool = True,
                      s_0: float = 4.0,
                      **kwargs) -> None:
        """
        Create and compile model.
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
        # default loss for classification
        if self.problem_type == 'classification':
            default_loss = self.loss_function
        # default loss for regression
        else:
            default_loss = 'mean_squared_error'
        kwargs['loss'] = kwargs.get('loss', default_loss)

        # default optimizer for classification
        if self.problem_type == 'classification':
            default_optimizer = 'adam'
        # default optimizer for regression
        else:
            default_optimizer = 'rmsprop'
        kwargs['optimizer'] = kwargs.get('optimizer', default_optimizer)

        # default metrics for classification
        if self.problem_type == 'classification':
            # default for binary classification
            if self.target_classes == 2:
                default_metrics = ['binary_accuracy']
            # default for multi-class classification
            else:
                default_metrics = ['categorical_accuracy']
        # default metrics for regression
        else:
            default_metrics = ['accuracy']
        kwargs['metrics'] = kwargs.get('metrics', default_metrics)

        # compile model and show model summary
        self.compile(**kwargs)

        # # initialize fuzzy rule centers
        # if init_c:
        #     self._initialize_centers(random=random)
        #
        # # initialize fuzzy rule widths
        # if init_s:
        #     self._initialize_widths(s_0=s_0)

        # print model summary
        print(self.summary())

    @staticmethod
    def loss_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    def get_layer_weights(self, layer: Union[str, int]) -> dict:
        """
        Get weights of layer based on input parameter.
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get weights from
            - input can be layer name or index
        """
        if isinstance(layer, int):
            return self.get_layer(index=layer).get_weights()
        else:
            return self.get_layer(name=layer).get_weights()

    def _initialize_centers(self, random: bool = True) -> None:
        """
        Initialize neuron center weights with samples from X_train dataset.

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
        self.get_layer(name='FuzzyRules').set_weights(start_weights)
        # validate weights updated as expected
        final_weights = self.get_layer_weights('FuzzyRules')
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
        c, s = self.get_layer_weights('FuzzyRules')

        # repeat s_0 value to array shaped like s
        s_init = np.repeat(s_0, s.size).reshape(s.shape)

        # set weights
        start_weights = [c, s_init]
        self.get_layer(name='FuzzyRules').set_weights(start_weights)
        # validate weights updated as expected
        final_weights = self.get_layer_weights('FuzzyRules')
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])
