import keras.api.ops as K
import numpy as np


def custom_loss_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
