import numpy as np
from library.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.prev_grad_weights = np.zeros_like(self.weights)
        self.prev_grad_biases = np.zeros_like(self.bias)
        self.step_sizes_weights = 0.1 * np.ones_like(self.weights)
        self.step_sizes_biases = 0.1 * np.ones_like(self.bias)
        pass

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate: float, use_rprop: bool):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        if use_rprop:
            self._rprop(weights_gradient, output_gradient)
        else:
            self.weights -= learning_rate * weights_gradient
            self.bias -= learning_rate * output_gradient
        return input_gradient
    
    def _rprop(self, weights_gradient, output_gradient):
        eta_plus = 1.2
        eta_minus = 0.5
        max_step = 50.0
        min_step = 1e-6
        
        change_w = np.sign(weights_gradient * self.prev_grad_weights)
        self.step_sizes_weights = np.where(change_w > 0, np.minimum(self.step_sizes_weights * eta_plus, max_step),
                                         np.where(change_w < 0, np.maximum(self.step_sizes_weights * eta_minus, min_step),
                                                  self.step_sizes_weights))
        weight_update = -np.sign(weights_gradient) * self.step_sizes_weights
        self.weights += weight_update
        self.prev_grad_weights = np.where(change_w < 0, 0, weights_gradient)
        
        # Bias updates
        change_b = np.sign(output_gradient * self.prev_grad_biases)
        self.step_sizes_biases = np.where(change_b > 0, np.minimum(self.step_sizes_biases * eta_plus, max_step),
                                        np.where(change_b < 0, np.maximum(self.step_sizes_biases * eta_minus, min_step),
                                                 self.step_sizes_biases))
        bias_update = -np.sign(output_gradient) * self.step_sizes_biases
        self.bias += bias_update
        self.prev_grad_biases = np.where(change_b < 0, 0, output_gradient)   