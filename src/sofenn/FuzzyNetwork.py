from typing import Optional

from keras.api.layers import Input, Dense
from keras.api.models import Model

from sofenn.callbacks import InitializeFuzzyWeights
from sofenn.layers import FuzzyLayer, NormalizeLayer, WeightedLayer, OutputLayer
from sofenn.losses import CustomLoss


# TODO: update to logging


class FuzzyNetwork(Model):
    """
    Fuzzy Network
    =============

    -Implemented per description in:
        "An on-line algorithm for creating self-organizing
        fuzzy neural networks" - Leng, Prasad, McGinnity (2004)
    -Composed of 5 layers with varying "fuzzy rule" nodes.

    Parameters
    ==========
    :param features: Number of features for input data.
    :param neurons: Number of neurons to use. Each one represents fuzzy neuron (if/then) operator.
    :param problem_type: Either 'classification' or 'regression' problem.
    :param target_classes: Optional number of classes in target data.
    :param name: Name of network (default: FuzzyNetwork).
    """
    def __init__(self,
                 features: int,
                 neurons: int = 1,
                 problem_type: str = 'classification',
                 target_classes: Optional[int] = 1,
                 name: str = 'FuzzyNetwork',
                 **kwargs):
        if features < 1:
            raise ValueError('At least 1 input feature required.')
        self.features = features

        if neurons <= 0:
            raise ValueError("Neurons must be a positive integer.")
        self.neurons = neurons

        if problem_type.lower() not in ['classification', 'regression']:
            raise ValueError(f"Invalid problem type provided: {problem_type}.")
        self.problem_type = problem_type.lower()

        if self.problem_type == 'classification':
            if target_classes is None:
                raise ValueError("Must provide target_classes parameter if 'problem_type' is 'classification'.")
            elif target_classes < 2:
                raise ValueError("Must specify more than 1 target class if 'problem_type' is 'classification'.")
        self.target_classes = None if self.problem_type == 'regression' else target_classes

        if 'name' not in kwargs:
            kwargs['name'] = name

        self.inputs = Input(name='Inputs', shape=(self.features,))
        self.fuzz = FuzzyLayer(shape=(self.features,), neurons=self.neurons)
        self.norm = NormalizeLayer(shape=(self.features, self.neurons))
        self.w = WeightedLayer(shape=[(self.features,), (self.neurons,)])
        self.raw = OutputLayer()
        self.softmax = Dense(self.target_classes, name='Softmax', activation='softmax')

        self.trained = False

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
        if self.problem_type == 'classification':
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
        if self.problem_type == 'classification':
            default_loss = CustomLoss
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

        super().compile(**kwargs)

    def fit(self, *args, **kwargs):
        if not self.built:
            print("FuzzyNetwork cannot be built until seeing training data.")

        kwargs['verbose'] = kwargs.get('verbose', 1)
        kwargs['epochs'] = kwargs.get('epochs', 100)
        kwargs['batch_size'] = kwargs.get('batch_size', 32)

        # add callback to instantiate fuzzy weights unless already provided
        x = kwargs['x'] if 'x' in kwargs else args[0]
        if 'callbacks' in kwargs:
            if any([isinstance(cb, InitializeFuzzyWeights) for cb in kwargs['callbacks']]):
                print('User already provided Fuzzy Weight Initializer callback.')
            else:
                kwargs['callbacks'].append(InitializeFuzzyWeights(sample_data=x))
        else:
            kwargs['callbacks'] = [InitializeFuzzyWeights(sample_data=x)]

        super().fit(*args, **kwargs)

        if not self.trained:
            self.trained = True


    def summary(self, *args, **kwargs):
        x = Input(shape=(self.features,), name="InputRow")
        model = Model(inputs=[x], outputs=self.call(x), name=self.name + ' Summary')
        return model.summary(*args, **kwargs)
