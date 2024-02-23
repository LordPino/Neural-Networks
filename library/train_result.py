
from typing import List

import numpy as np

from library.types import ActivationFunction, OutputFunction


class TrainResult:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, w: List[np.ndarray], b: List[np.ndarray], activation_functions: List[ActivationFunction], output_function: OutputFunction):
        self._x_train = x_train
        self._y_train = y_train
        self._w = w
        self._b = b

        self._activation_functions = activation_functions
        self._output_function = output_function
        pass

    def get_weights(self):
        return self._w
    
    def get_biases(self):
        return self._b
    
    def get_x_train(self):
        return self._x_train
    
    def get_y_train(self):
        return self._y_train
    
    def get_activation_functions(self) -> List[ActivationFunction]:
        return self._activation_functions
    
    def get_output_function(self) -> OutputFunction:
        return self._output_function