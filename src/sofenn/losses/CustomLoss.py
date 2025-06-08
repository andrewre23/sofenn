import keras.api.ops as K
from keras.src.losses import Loss
import keras

# def custom_loss_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     """
#     Custom loss function
#
#     E = exp{-sum[i=1,j; 1/2 * [pred(j) - test(j)]^2]}
#
#     Parameters
#     ==========
#     y_true : np.array
#         - true values
#     y_pred : np.array
#         - predicted values
#     """
#     return K.sum(1 / 2 * K.square(y_pred - y_true))


@keras.saving.register_keras_serializable()
class CustomLoss(Loss):
    """
    Custom loss function.

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
        return 1 / 2 * K.square(y_pred - y_true)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
