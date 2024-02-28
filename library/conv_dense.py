import numpy as np
from library.layer import Layer

class ConvDense(Layer):
    def __init__(self, input_size, kernal_number):
        self.k = np.random.randn(input_size[1], kernal_number)
        
        self.biases = np.random.randn(input_size[0], 1)
        
        self.prev_grad_k = np.zeros_like(self.k)
        self.prev_grad_biases = np.zeros_like(self.biases)
        self.step_sizes_k = 0.1 * np.ones_like(self.k)
        self.step_sizes_biases = 0.1 * np.ones_like(self.biases)
        pass

    def forward(self, input):
        self.input = input
        
        partial_output = np.dot(self.input, self.k)
        output = np.sum(partial_output, axis=1)
        output = output[:, np.newaxis]
        output += self.biases
        
        return output

    def backward(self, output_gradient, learning_rate: float, use_rprop: bool):
        k_gradient = np.dot(output_gradient.T, self.input).T
        # problema delle dimensioni causa somma 
        input_gradient = np.dot(output_gradient, self.k.T)
        
        if not use_rprop:
            self.k -= learning_rate * k_gradient
            self.biases -= learning_rate * output_gradient
        else:
            self._rprop(k_gradient, output_gradient)
        return input_gradient
    
    def _rprop(self, k_gradient, output_gradient):
        eta_plus = 1.2
        eta_minus = 0.5
        max_step = 50.0
        min_step = 1e-6
        
        change_w = np.sign(k_gradient * self.prev_grad_k)
        self.step_sizes_k = np.where(change_w > 0, np.minimum(self.step_sizes_k * eta_plus, max_step),
                                         np.where(change_w < 0, np.maximum(self.step_sizes_k * eta_minus, min_step),
                                                  self.step_sizes_k))
        k_update = -np.sign(k_gradient) * self.step_sizes_k
        self.k += k_update
        self.prev_grad_k = np.where(change_w < 0, 0, k_gradient)
        
        # Bias updates
        change_b = np.sign(output_gradient * self.prev_grad_biases)
        self.step_sizes_biases = np.where(change_b > 0, np.minimum(self.step_sizes_biases * eta_plus, max_step),
                                        np.where(change_b < 0, np.maximum(self.step_sizes_biases * eta_minus, min_step),
                                                 self.step_sizes_biases))
        bias_update = -np.sign(output_gradient) * self.step_sizes_biases
        self.biases += bias_update
        self.prev_grad_biases = np.where(change_b < 0, 0, output_gradient)
        
        
        