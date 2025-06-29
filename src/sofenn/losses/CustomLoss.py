import logging

import keras
import keras.ops as k
from keras.losses import Loss

logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable()
class CustomLoss(Loss):
    """
    Custom Loss Function
    ====================
    Measures loss as half of the square of the error.

    E = exp{-sum[i=1,j; 1/2 * [pred(j) - test(j)]^2]}
    """
    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'CustomLoss'
        if 'reduction' not in kwargs:
            kwargs['reduction'] = 'sum'
        super(CustomLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        Call the custom loss function.

        :param y_true: True values.
        :param y_pred: Predicted values.
        """
        return 1 / 2 * k.square(y_pred - y_true)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
