import numpy as np

from library.types import ActivationDerivate, ActivationFunction, OutputDerivate, OutputFunction



class Layer:
    def __init__(self, neurons: int, activation_function: ActivationFunction, activation_derivate: ActivationDerivate):
        self._neurons = neurons
        self._activation_function = activation_function
        self._activation_derivate = activation_derivate
        self._weights = None
        self._biases = None

    def get_neurons(self) -> int:
        return self._neurons

    def get_weights(self) -> np.ndarray:
        return self._weights
    
    def set_weights(self, weights: np.ndarray):
        self._weights = weights

    def get_biases(self) -> np.ndarray:
        return self._biases
    
    def set_biases(self, biases: np.ndarray):
        self._biases = biases

    def get_activation_function(self) -> ActivationFunction:
        return self._activation_function

    def get_activation_derivate(self) -> ActivationDerivate:
        return self._activation_derivate

class OutputLayer(Layer):
    def __init__(self, neurons: int, output_function: OutputFunction, output_derivate: OutputDerivate):
        super().__init__(neurons, output_function, output_derivate)
    
    def get_output_function(self):
        return self.get_activation_function()

    def get_output_derivate(self):
        return self.get_activation_derivate()