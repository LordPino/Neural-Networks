
import numpy as np
from typing import List

from library.layer import Layer

class RPropParams:
    def __init__(self, layers: List[Layer]):
        weights = [l.get_weights() for l in layers]
        biases = [l.get_biases() for l in layers]
        self._step_sizes_weights = [0.1 * np.ones_like(w) for w in weights]
        self._step_sizes_biases = [0.1 * np.ones_like(b) for b in biases]

        # Initialize previous gradients to zero (for weights and biases)
        self._prev_grad_weights = [np.zeros_like(w) for w in weights]
        self._prev_grad_biases = [np.zeros_like(b) for b in biases]
        pass

    def get_step_sizes_weights(self):
        return self._step_sizes_weights
    
    def get_step_sizes_biases(self):
        return self._step_sizes_biases
    
    def get_prev_grad_weights(self):
        return self._prev_grad_weights
    
    def get_prev_grad_biases(self):
        return self._prev_grad_biases