
from library.layer import Layer
import numpy as np
from typing import Tuple

class Network:
    def __init__(self, epochs: int):
        self._layers = list[Layer]
        self._output_layer = None
        self._output_function = None
        self._epochs = epochs
        pass

    def is_valid(self):
        return len(self._layers) < 2
        
    def add_layer(self, layer: Layer):
        self._layers.append(layer)
        pass

    def get_output_layer(self) -> Layer:
        if not self.is_valid():
            raise ValueError("Network: number of layers must be atleast 2.")
        return self._layers[-1]

    def get_num_layers(self) -> int:
        return len(self._layers)

    # Call after adding all layers
    def init_layers(self):
        if not self.is_valid():
            raise ValueError("Network: number of layers must be atleast 2.")
        
        first_layer = self._layers[0]
        first_layer.set_weight(np.random.rand(self._layers[1].get_neurons(), first_layer.get_neurons())  - 0.5) 
        first_layer.set_bias(np.random.rand(self._layers[1].get_neurons(), 1) - 0.5)

        # Middle layers and output layer
        for i in range(1, len(self._layers) - 1):
            layer = self._layers[i]

            layer.set_weight(np.random.rand(layer.get_neurons(), self._layers[i-1].get_neurons()) - 0.5)
            layer.set_bias(np.random.rand(layer.get_neurons(), 1) - 0.5)
        
        # Output layer
        output_layer = self.get_output_layer()
        output_layer.set_weight(np.random.rand(output_layer.get_neurons(), self._layers[-2].get_neurons()) - 0.5)
        output_layer.set_bias(np.random.rand(output_layer.get_neurons(), 1) - 0.5)

    def gradint_descent(self, X: np.ndarray, Y: np.ndarray):
        self.init_layers()

        for i in range(self._epochs):
            a, z = self._forward_propagation(x=X)

        pass

    def _forward_propagation(self, x: np.ndarray) -> Tuple[list[np.ndarray], list[np.ndarray]]: 
        num_layers = self.get_num_layers()

        # Preallocate lists for z and a with the appropriate sizes
        z = [None] * num_layers
        a = [None] * (num_layers + 1)  # +1 for the input layer

        a[0] = x

        # Iterate through each layer except the last one
        for i in range(num_layers - 1):
            layer = self._layers[i]

            z[i] = np.dot(layer.get_weight(), a[i]) + layer.get_bias()
            a[i + 1] = layer.activation_function(z[i])

        # Handle the output layer separately
        output_layer = self.get_output_layer()
        z[-1] = np.dot(output_layer.get_weight(), a[-2]) + output_layer.get_bias()
        a[-1] = self._output_function(z[-1])

        return a, z

    def _back_propagation(self, y: np.ndarray):
        pass