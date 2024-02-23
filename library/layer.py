import numpy as np
from typing import Callable, List

ActivationFunction = Callable[[np.ndarray], np.ndarray]

class Layer:
    def __init__(self, neurons: int, activation_function: list[Callable[[np.ndarray], np.ndarray]]):
        self._neurons = neurons
        self.activation_function = activation_function
        self._weights = []
        self._biases = []

    def get_neurons(self) -> int:
        return self._neurons

    def get_weights(self) -> list[int]:
        return self._weights
    
    def set_weights(self, weights: list[float]):
        self._weights = weights

    def get_biases(self) -> list[int]:
        return self._biases
    
    def set_bias(self, biases: list[float]):
        self._biases = biases

    def get_activation_function(self) -> ActivationFunction:
        return self.activation_function

class OutputLayer(Layer):
    def __init__(self, neurons: int, output_function):
        super.__init__(neurons, output_function)
    
    def get_output_function(self):
        return self.activation_function