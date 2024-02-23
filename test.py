
import numpy as np
from library.layer import Layer, OutputLayer
from library.network import FunctionError, Network
from library.utils import ReLU_derivative, soft_max_derivative

def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def soft_max(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / sum(np.exp(z))

network = Network()

layer1 = Layer(neurons=784, activation_function=ReLU, activation_derivate=ReLU_derivative)
layer2 = OutputLayer(neurons=10, utput_function=soft_max, output_derivate=soft_max_derivative)

network.use_rprop(True).use_softmax(True).error_function(FunctionError.MSE).add_layer(layer1).add_layer(layer2)

network.train()