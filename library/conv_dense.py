import numpy as np
from library.layer import Layer
from library.utils import flatten, matrix_multiply_and_flatten, reconstruct_from_flattened_patches, reshape_columns_to_square_matrices

class ConvDense(Layer):
    def __init__(self, input_size, kernal_number, use_cross_entropy: bool = False):
        self.input_size = input_size
        self.k = np.random.randn(input_size[1], kernal_number)
        self.biases = np.random.randn(input_size[0], kernal_number)
        
        self.prev_grad_k = np.zeros_like(self.k)
        self.prev_grad_biases = np.zeros_like(self.biases)
        self.step_sizes_k = 0.1 * np.ones_like(self.k)
        self.step_sizes_biases = 0.1 * np.ones_like(self.biases)
        pass

    def forward(self, input):
        self.input = input
    
        output = np.dot(self.input, self.k) + self.biases
        return output

    '''
    def backward(self, output_gradient, learning_rate: float, use_rprop: bool):
        k_gradient = np.dot(self.input.T, output_gradient)
        # problema delle dimensioni causa somma 
        input_gradient = np.dot(output_gradient, self.k.T)
        
        if not use_rprop:
            self.k -= learning_rate * k_gradient
            self.biases -= learning_rate * output_gradient
        else:
            self._rprop(k_gradient, output_gradient)
        return input_gradient
    '''
    
    def backward(self, output_gradient, learning_rate: float, use_rprop: bool):
        y_list = reshape_columns_to_square_matrices(output_gradient)
        
        x = reconstruct_from_flattened_patches(self.input, 28, 28, int(np.sqrt(self.input.shape[1])), int(np.sqrt(self.input.shape[1])))
        
        stride = (x.shape[0] - y_list[0].shape[0]) // int((np.sqrt(self.input_size[1])) - 1)
        m_new = flatten(x, stride, y_list[0].shape[0])
        m_new = np.array(m_new)
        
        dk = matrix_multiply_and_flatten(m_new, y_list)
        
        if not use_rprop:
            self.biases -= learning_rate * output_gradient
            self.k -= learning_rate * dk
        else:
            self._rprop(dk, output_gradient)
        
        return self.input
        
    def _rprop(self, k_gradient, output_gradient):
        eta_plus = 1.2
        eta_minus = 0.5
        max_step = 50.0
        min_step = 1e-6
        
        bias_gradient = output_gradient.sum(axis=0, keepdims=True)
        # Update for weights
        for param, delta, prev_grad, grad in [
            (self.k, self.step_sizes_k, self.prev_grad_k, k_gradient),
            (self.biases, self.step_sizes_biases, self.prev_grad_biases, bias_gradient)
        ]:
            sign_change = np.sign(grad) * np.sign(prev_grad)
            delta *= np.where(sign_change > 0, eta_plus, np.where(sign_change < 0, eta_minus, 1))
            delta = np.clip(delta, min_step, max_step)

            update_direction = np.sign(grad)
            param -= update_direction * delta
            # Only reset gradients when sign changes
            np.copyto(prev_grad, grad, where=sign_change >= 0)

        # Ensure the updates for next iteration are prepared
        self.prev_grad_k = k_gradient
        self.prev_grad_biases = bias_gradient
            
            