import numpy as np
from library.dense import Dense
from scipy import signal

class Convolutional(Dense):
    def __init__(self, input_shape, kernel_size, depth, flatten = False):
        input_depth, input_height, input_width = input_shape
        
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        self._flatten = flatten
        
        Dense.__init__(self, input_depth * input_height * input_width, depth* (input_height - kernel_size + 1) * (input_width - kernel_size + 1))
        
    def forward(self, input):
        if self._flatten:
            _, _, kernel_size = self.kernels_shape
            self.input = self.flatten(input,1, kernel_size)
            return np.dot(self.input, self.weights) + self.bias
        else:
            self.input = input
            self.output = np.copy(self.biases)
            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
            return self.output

    def backward(self, output_gradient, learning_rate: float, use_rprop: bool):
        if self._flatten:
            # TODO: change
            return Dense.backward(self, output_gradient, learning_rate, use_rprop)
        else:
            kernels_gradient = np.zeros(self.kernels_shape)
            input_gradient = np.zeros(self.input_shape)

            for i in range(self.depth):
                for j in range(self.input_depth):
                    kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                    input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

            self.kernels -= learning_rate * kernels_gradient
            self.biases -= learning_rate * output_gradient
            return input_gradient

    def flatten(self, input, stride, kernel_size):
        flattened_patches = []
        for i in range(0, input.shape[0] - kernel_size + 1, stride):
            for j in range(0, input.shape[1] - kernel_size + 1, stride):
                # Extract the patch
                patch = input[i:i+kernel_size, j:j+kernel_size]
                # Flatten and add the patch to the list
                flattened_patches.append(patch.flatten())
        return flattened_patches