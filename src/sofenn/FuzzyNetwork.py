from typing import Union, Optional

import numpy as np
from keras.api.layers import Input, Dense
from keras.api.models import Model

from sofenn.callbacks import InitializeFuzzyWeights
from sofenn.layers import FuzzyLayer, NormalizeLayer, WeightedLayer, OutputLayer
from sofenn.losses import custom_loss_function


# TODO: update to logging


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
                 features: int,
                 neurons: int = 1,
                 max_neurons: int = 100,
                 prob_type: str = 'classification',
                 target_classes: Optional[int] = 1,
                 name: str = 'FuzzyNetwork',
                 debug: bool = True,
                 **kwargs):

        # set debug flag
        self._debug = debug

        if features <1:
            raise ValueError('FuzzyNetwork requires at least 1 input feature.')
        self.features = features

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
        if 'name' not in kwargs:
            kwargs['name'] = name
        self.inputs = Input(name='Inputs', shape=(self.features,))
        self.fuzz = FuzzyLayer(shape=(self.features,), neurons=self.neurons)
        self.norm = NormalizeLayer(shape=(self.features, self.neurons))
        self.w = WeightedLayer(shape=[(self.features,), (self.neurons,)])
        self.raw = OutputLayer()
        self.softmax = Dense(self.target_classes, name='Softmax', activation='softmax')

        super().__init__(**kwargs)

    def call(self, inputs):
        """
        Call model.
        Note: keras.Model subclasses are not built until executing '.call()', as the model requires
        visibility into the input data to properly initialize.

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
        phi = self.fuzz(inputs)
        psi = self.norm(phi)
        f = self.w([inputs, psi])
        raw_output = self.raw(f)
        final_out = raw_output

        # add softmax layer for classification problem
        if self.prob_type == 'classification':
            classes = self.softmax(raw_output)
            final_out = classes

        # TODO: determine logic for activation w/ regression
        # # extract activation from kwargs
        # if 'activation' not in kwargs:
        #     activation = 'sigmoid'
        # else:
        #     activation = kwargs['activation']
        # preds = Activation(name='OutputActivation', activation=activation)(raw_output)

        return final_out

    def compile(self, **kwargs) -> None:
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

        if self.prob_type == 'classification':
            default_loss = custom_loss_function
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
        super().compile(**kwargs)

        # print model summary
        if self._debug:
            print(self.summary())


    def fit(self, *args, **kwargs):
        if not self.built:
            print("FuzzyNetwork cannot be built until seeing training data.")

        # set default verbose setting
        default_verbose = 1
        kwargs['verbose'] = kwargs.get('verbose', default_verbose)

        # set default training epochs
        default_epochs = 100
        kwargs['epochs'] = kwargs.get('epochs', default_epochs)

        # set default training epochs
        default_batch_size = 32
        kwargs['batch_size'] = kwargs.get('batch_size', default_batch_size)

        # add callback to instantiate fuzzy weights unless already added
        x = kwargs['x'] if 'x' in kwargs else args[0]
        if 'callbacks' in kwargs:
            if any([isinstance(cb, InitializeFuzzyWeights) for cb in kwargs['callbacks']]):
                print('User already provided Fuzzy Weight Initializer callback.')
            else:
                kwargs['callbacks'].append(InitializeFuzzyWeights(sample_data=x))
        else:
            kwargs['callbacks'] = [InitializeFuzzyWeights(sample_data=x)]

        # fit model to dataset
        super().fit(*args, **kwargs)
