
import numpy as np
from library.layer import Layer
from library.network import Network

def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def soft_max(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / sum(np.exp(z))

network = Network()

layer1 = Layer(784, ReLU)
layer2 = Layer(10, soft_max)

network.add_layer(layer1)
network.add_layer(layer2)

network.train()