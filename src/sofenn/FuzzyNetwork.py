from typing import Union, Optional

import keras
import keras.api.ops as K
import numpy as np
from keras.api.layers import Input, Dense
from keras.api.layers import Layer
from keras.api.models import Model
from keras.api.utils import to_categorical
from sklearn.metrics import mean_absolute_error

from sofenn.layers import FuzzyLayer, NormalizeLayer, WeightedLayer, OutputLayer

# TODO: remove X/y train/test as inputs and replace with both input_tensor and input_shape as optional inputs
class FuzzyNetworkModel(object):
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
                 input_tensor: Optional[keras.KerasTensor] = None,
                 input_shape: Optional[tuple] = None,
                 neurons: int = 1,
                 max_neurons: int = 100,
                 prob_type: str = 'classification',
                 target_classes: Optional[int] = 1,
                 debug: bool = True,
                 **kwargs):

        # set debug flag
        self._debug = debug

        # validate input tensor / shape
        if input_tensor is None and input_shape is None:
            raise ValueError("Must provide at least one of 'input_tensor' or 'input_shape'.")
        elif input_tensor and input_shape:
            if input_tensor.shape != input_shape:
                raise ValueError(f"Input tensor's shape must match 'input_shape' parameter if both provided. "
                                 f"input tensor's shape: {input_tensor.shape}. "
                                 f"'input_shape' parameter: {input_shape}.")
        self.features = input_shape[-1]

        # set output problem type
        if prob_type.lower() not in ['classification', 'regression']:
            raise ValueError("Invalid problem type.")
        elif prob_type.lower() == 'classification' and target_classes is None:
            raise ValueError("Must provide target_classes parameter if 'prob_type' is 'classification'.")
        elif prob_type.lower() == 'classification' and target_classes < 2:
            raise ValueError("Must specify more than 1 target class if 'prob_type' is 'classification'.")
        self.prob_type = prob_type
        self.target_classes = target_classes

        # set neuron attributes
        # initial number of neurons
        if type(neurons) is not int or neurons <= 0:
            raise ValueError("Must enter positive integer.")
        self.neurons = neurons

        # max number of neurons
        if type(max_neurons) is not int or max_neurons < neurons:
            raise ValueError("Must enter positive integer no less than number of neuron.s")
        self.max_neurons = max_neurons

        # define model and set model attribute
        self.model = None
        self.build_model(**kwargs)

    def build_model(self, **kwargs) -> None:
        """
        Build and initialize Model.

        Layers
        ======
        0 - Input Layer
                input dataset
            - input shape  : (*, features)
        1 - Radial Basis Function Layer (Fuzzy Layer)
                layer to hold fuzzy rules for complex system
            - input : x
                shape: (*, features)
            - output : phi
                shape : (*, neurons)
        2 - Normalize Layer
                normalize each output of previous layer as
                relative amount from sum of all previous outputs
            - input : phi
                shape  : (*, neurons)
            - output : psi
                shape : (*, neurons)
        3 - Weighted Layer
                multiply bias vector (1+n_features, neurons) by
                parameter vector (1+n_features,) of parameters
                from each fuzzy rule
                multiply each product by output of each rule's
                layer from normalize layer
            - inputs : [x, psi]
                shape  : [(*, 1+features), (*, neurons)]
            - output : f
                shape : (*, neurons)
        4 - Output Layer
                summation of incoming signals from weighted layer
            - input shape  : (*, neurons)
            - output shape : (*,)

        5 - Softmax Layer (classification)
            softmax layer for classification problems
            - input shape : (*, 1)
            - output shape : (*, classes)
        """
        if self._debug:
            print('Building Fuzzy Network with {} neurons...'
                  .format(self.neurons))

        # add layers
        inputs = Input(name='Inputs', shape=(self.features,))
        fuzz = FuzzyLayer(shape=(self.features,), neurons=self.neurons)
        norm = NormalizeLayer(shape=(self.features, self.neurons))
        weights = WeightedLayer(shape=[(self.features,), (self.neurons,)])
        raw = OutputLayer()

        # run through layers
        phi = fuzz(inputs)
        psi = norm(phi)
        f = weights([inputs, psi])
        raw_output = raw(f)
        final_out = raw_output
        # add softmax layer for classification problem
        if self.prob_type == 'classification':
            classify = Dense(self.target_classes,
                            name='Softmax', activation='softmax')
            classes = classify(raw_output)
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

    def compile_model(self,
                      init_c: bool = True,
                      sample_data: Optional[np.ndarray] = None,
                      random_sample: bool = True,
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
        sample_data : np.ndarray
            - sample data to initialize centers
        random_sample : bool
            - take either random samples or first samples that appear in training data
        init_s : bool
            - run method to initialize widths or take default initializations
        s_0 : float
            - value for initial centers of neurons
        """
        if self._debug:
            print('Compiling model...')

        if init_c and sample_data is None:
            raise ValueError('Must provide sample data when initializing centers.')

        if self.prob_type == 'classification':
            default_loss = self.loss_function
            default_optimizer = 'adam'
            default_metrics = ['categorical_accuracy']
            # if self.y_test.ndim == 2:                       # binary classification
            #     default_metrics = ['binary_accuracy']
            # else:                                           # multi-class classification
            #     default_metrics = ['categorical_accuracy']
        else:
            default_loss = 'mean_squared_error'
            default_optimizer = 'rmsprop'
            default_metrics = ['accuracy']
        kwargs['loss'] = kwargs.get('loss', default_loss)
        kwargs['optimizer'] = kwargs.get('optimizer', default_optimizer)
        kwargs['metrics'] = kwargs.get('metrics', default_metrics)

        # compile model and show model summary
        self.model.compile(**kwargs)

        # initialize fuzzy rule centers
        if init_c:
            self._initialize_centers(sample_data=sample_data, random_sample=random_sample)

        # initialize fuzzy rule widths
        if init_s:
            self._initialize_widths(s_0=s_0)

        # print model summary
        if self._debug:
            print(self.model.summary())

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

    def train_model(self, x, y, **kwargs) -> None:
        """
        Fit model on current training data.
        """
        if self._debug:
            print('Training model...')

        # set default verbose setting
        default_verbose = 1
        kwargs['verbose'] = kwargs.get('verbose', default_verbose)

        # set default training epochs
        default_epochs = 100
        kwargs['epochs'] = kwargs.get('epochs', default_epochs)

        # set default training epochs
        default_batch_size = 32
        kwargs['batch_size'] = kwargs.get('batch_size', default_batch_size)

        # fit model to dataset
        self.model.fit(x, y, **kwargs)

    def get_layer(self, layer: Union[str, int]) -> Layer:
        """
        Get layer object based on input parameter.
            - exception of Input layer

        Parameters
        ==========
        layer : str or int
            - layer to get weights from
            - input can be layer name or index
        """
        # if named parameter
        if layer in [mlayer.name for mlayer in self.model.layers]:
            layer_out = self.model.get_layer(layer)

        # if indexed parameter
        elif layer in range(len(self.model.layers)):
            layer_out = self.model.layers[layer]

        else:
            raise ValueError('Error: layer must be layer name or index')

        return layer_out

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
        return self.get_layer(layer).get_weights()

    def get_layer_output(self, layer: Union[str, int]) -> np.ndarray:
        """
        Get output of layer based on input parameter.
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
            samples = np.random.randint(0, len(sample_data), self.neurons)
            x_i = np.array([sample_data[samp] for samp in samples])
        else:
            # take first few samples, one for each neuron
            x_i = sample_data[:self.neurons]

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
        self.get_layer('FuzzyRules').set_weights(start_weights)
        # validate weights updated as expected
        final_weights = self.get_layer_weights('FuzzyRules')
        assert np.allclose(start_weights[0], final_weights[0])
        assert np.allclose(start_weights[1], final_weights[1])
