import numpy as np
from library.layer import Layer


class InputLayer(Layer):
    def __init__(self, kernel_number: int):
        super().__init__()
        self.kernel_number = kernel_number

    def forward(self, input):
        output = np.tile(input, (1, self.kernel_number))
        return output

    def backward(self, output_gradient, learning_rate: float, use_rprop: bool):
        width_shape = output_gradient.shape[1] // self.kernel_number
        output = output_gradient[:, :width_shape]        
        return output

    def _rprop(self, k_gradient, output_gradient):
        pass