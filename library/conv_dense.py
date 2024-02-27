import numpy as np
from library.layer import Layer

class ConvDense(Layer):
    def __init__(self, input_size, kernal_number):
        self.k = np.random.randn(input_size[1], kernal_number)
        
        self.bias = np.random.randn(input_size[0], kernal_number)
        
        # TODO(): rprop in conv dense
        """
        self.prev_grad_weights = np.zeros_like(self.weights)
        self.prev_grad_biases = np.zeros_like(self.bias)
        self.step_sizes_weights = 0.1 * np.ones_like(self.weights)
        self.step_sizes_biases = 0.1 * np.ones_like(self.bias)
        """
        pass

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.k) + self.bias

    def backward(self, output_gradient, learning_rate: float, use_rprop: bool):
        k_gradient = np.dot(output_gradient.T, self.input).T
        input_gradient = np.dot(output_gradient, self.k.T)
        
        # print("output shape: " + output_gradient.shape)
        # print("self.input shape: " + self.input.shape)
        # print("k_gradient shape: " + k_gradient.shape)
        # print("input_gradient: " + input_gradient.shape)
        
        self.k -= learning_rate * k_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
        
    
    def _rprop(self, weights_gradient, output_gradient):
        eta_plus = 1.2
        eta_minus = 0.5
        max_step = 50.0
        min_step = 1e-6
        
        change_w = np.sign(weights_gradient * prev_weights_gradient)
        self.step_sizes_weights = np.where(change_w > 0, np.minimum(self.step_sizes_weights * eta_plus, max_step),
                                         np.where(change_w < 0, np.maximum(self.step_sizes_weights * eta_minus, min_step),
                                                  self.step_sizes_weights))
        weight_update = -np.sign(weights_gradient) * self.step_sizes_weights
        self.weights += weight_update
        prev_weights_gradient = np.where(change_w < 0, 0, weights_gradient)
        
        # Bias updates
        change_b = np.sign(output_gradient * self.prev_grad_biases)
        self.step_sizes_biases = np.where(change_b > 0, np.minimum(self.step_sizes_biases * eta_plus, max_step),
                                        np.where(change_b < 0, np.maximum(self.step_sizes_biases * eta_minus, min_step),
                                                 self.step_sizes_biases))
        bias_update = -np.sign(output_gradient) * self.step_sizes_biases
        self.biases += bias_update
        self.prev_grad_biases = np.where(change_b < 0, 0, output_gradient)
        
        
        