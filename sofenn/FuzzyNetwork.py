#
# SOFENN
# Self-Organizing Fuzzy Neural Network
#
# (sounds like soften)
#
#
# Implemented per description in
# An on-line algorithm for creating self-organizing
# fuzzy neural networks
# Leng, Prasad, McGinnity (2004)
#
#
# Andrew Edmonds - 2019
# github.com/andrewre23
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Activation

from sklearn.metrics import confusion_matrix, classification_report, \
    mean_absolute_error, roc_auc_score

# custom Fuzzy Layers
from .layers import FuzzyLayer, NormalizedLayer, WeightedLayer, OutputLayer


class FuzzyNetwork(Model):
    def __init__(self):
        pass
