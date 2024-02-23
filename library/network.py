
from enum import Enum
from library.layer import Layer, OutputLayer
import numpy as np
from typing import List, Tuple
from library.train_result import TrainResult
from library.types import ActivationFunction, OutputFunction

from library.utils import derivatie_cross_entropy, derivative_mse, function_derivative, get_accuracy, get_predictions, one_hot

class FunctionError(Enum):
    MSE = 1 # Mean Squared Error
    CROSS_ENTROPY = 2 # Cross Entropy Loss

class Network:
    def __init__(self, epochs: int):
        self._layers = list[Layer]()
        self._output_layer = None
        self._output_function = None
        self._epochs = epochs
        self._use_rprop = False
        self._use_softmax = False
        self._error_function = None
        self._learning_rate = 0.1
        pass
    
    def train(self, X: np.ndarray, Y: np.ndarray) -> TrainResult:
        W, B = self._gradint_descent(X=X, Y=Y)

        return TrainResult(
            x_train=X,
            y_train=Y,
            w=W,
            b=B,
            activation_functions=[l.get_activation_function() for l in self._layers],
            output_function=self.get_output_layer().get_output_function()
        )



    def make_predictions(self, x: np.ndarray) -> np.ndarray:
        a, _ = self._forward_propagation(x=x)
        return get_predictions(a[-1])

    def is_valid(self):
        return len(self._layers) < 2
        
    def add_layer(self, layer: Layer):
        if(layer is OutputLayer):
            if(self._layers[-1] is OutputLayer):
                raise ValueError('[Network]: Cannot add an Output Layer twice.')
            
        self._layers.append(layer)
        return self

    def remove_layer(self, index: int) -> Layer:
        if(index < 0 or index >= len(self._layers)):
            raise ValueError("[Network]: Layer index out of bounds.")
        layer = self._layers[index]
        del self._layers[index]
        return layer

    def error_function(self, error: FunctionError):
        self._error_function = error
        return self

    def use_rprop(self, v: bool):
        self._use_rprop = v
        return self

    def is_using_rprop(self) -> bool:
        return self._use_rprop

    def learning_rate(self, value: float):
        self._learning_rate = value
        return self

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def use_softmax(self, v: bool):
        self._use_softmax = v
        return self

    def is_using_softmax(self) -> bool:
        return self._use_softmax

    def get_output_layer(self) -> OutputLayer:
        if not self.is_valid():
            raise ValueError("[Network]: number of layers must be atleast 2.")
        last_layer = self._layers[-1]

        if(last_layer is not OutputLayer):
            raise ValueError("[Network]: There's no Output layer.")

        return last_layer

    def get_num_layers(self) -> int:
        return len(self._layers)

    # Call after adding all layers
    def init_layers(self):
        if not self.is_valid():
            raise ValueError("[Network]: number of layers must be atleast 2.")
        
        first_layer = self._layers[0]
        first_layer.set_weights(np.random.rand(self._layers[1].get_neurons(), first_layer.get_neurons())  - 0.5) 
        first_layer.set_biases(np.random.rand(self._layers[1].get_neurons(), 1) - 0.5)

        # Middle layers and output layer
        for i in range(1, len(self._layers) - 1):
            layer = self._layers[i]

            layer.set_weights(np.random.rand(layer.get_neurons(), self._layers[i-1].get_neurons()) - 0.5)
            layer.set_biases(np.random.rand(layer.get_neurons(), 1) - 0.5)
        
        # Output layer
        output_layer = self.get_output_layer()
        output_layer.set_weights(np.random.rand(output_layer.get_neurons(), self._layers[-2].get_neurons()) - 0.5)
        output_layer.set_biases(np.random.rand(output_layer.get_neurons(), 1) - 0.5)

    def _gradint_descent(self, X: np.ndarray, Y: np.ndarray):
        self.init_layers()

        if self._use_rprop:
            step_sizes_weights, step_sizes_biases, prev_grad_weights, prev_grad_biases = self._init_rprop_params(weights=weights, biases=biases)

        
        for i in range(self._epochs):
            a, z = self._forward_propagation(x=X)
            dW, dB = self._back_propagation(y=Y, z=z, a=a)

            
            if self._use_rprop:
                weights, biases, prev_grad_weights, prev_grad_biases, step_sizes_weights, step_sizes_biases = self._rprop_update(weights=weights, 
                                                                                                                       biases=biases, 
                                                                                                                       grad_weights=dW,
                                                                                                                       grad_biases=dB,
                                                                                                                       step_sizes_weights=step_sizes_weights, 
                                                                                                                       step_sizes_biases=step_sizes_biases, 
                                                                                                                       prev_grad_weights=prev_grad_weights, 
                                                                                                                       prev_grad_biases=prev_grad_biases)
            else:
                weights, biases = self._update_params(weights=weights, 
                    biases=biases, 
                    dW=dW, 
                    dB=dB)
            
            if i % 10 == 0:
                print("epoch: ", i)
                print("Accuracy: ", get_accuracy(get_predictions(a[-1]), Y))
        return weights, biases

    def _forward_propagation(self, x: np.ndarray) -> Tuple[list[np.ndarray], list[np.ndarray]]: 
        num_layers = self.get_num_layers()

        # Preallocate lists for z and a with the appropriate sizes
        z = [] * num_layers
        a = [] * (num_layers + 1)  # +1 for the input layer

        a[0] = x

        # Iterate through each layer except the last one
        for i in range(num_layers - 1):
            layer = self._layers[i]

            z[i] = np.dot(layer.get_weights(), a[i]) + layer.get_biases()
            a[i + 1] = layer.get_activation_function(z[i])

        # Handle the output layer separately
        output_layer = self.get_output_layer()
        z[-1] = np.dot(output_layer.get_weights(), a[-2]) + output_layer.get_biases()
        a[-1] = output_layer.get_output_function(z[-1])

        return a, z

    def _back_propagation(self, y: np.ndarray, z: List[np.ndarray], a: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        m = y.shape[0]
        one_hot_y = one_hot(y)

        
        weights = [l.get_weights() for l in self._layers]
        biases = [l.get_biases() for l in self._layers]
        dW = [] * len(weights)
        dB = [] * len(biases)

        # Output layer error
        if self._error_function == FunctionError.MSE:
            dZ = derivative_mse(one_hot_y, a[-1])
        elif self._error_function == FunctionError.CROSS_ENTROPY:
            if self._use_softmax:
                dZ = a[-1] - one_hot_y
            else:
                dZ = derivatie_cross_entropy(one_hot_y, a[-1])

        if self._error_function == FunctionError.MSE or (self._error_function == FunctionError.CROSS_ENTROPY and self._use_softmax is False):
            output_layer = self.get_output_layer()
            output_derivative = output_layer.get_output_derivate()
            if output_derivative is not None:
                dZ = dZ * output_derivative(a[-1])
            else:
                dZ = dZ * function_derivative(output_layer.get_output_function(), a[-1])

        # Output layer gradients
        dW[-1] = (1/m) * np.dot(dZ, a[-2].T)
        dB[-1] = (1/m) * np.sum(dZ)
        
        # Backpropagation through layers
        for i in range(len(weights) - 2, -1, -1):
            dA = np.dot(weights[i + 1].T, dZ)
            layer = self._layers[i]
            if layer.get_activation_derivate() is not None:
                dZ = dA * layer.get_activation_derivate()(z[i])
            else:
                dZ = dA * function_derivative(layer.get_activation_function(), z[i])
            dW[i] = (1/m) * np.dot(dZ, a[i].T)
            dB[i] = (1/m) * np.sum(dZ)

        return dW, dB
    
    # Init step sizes and gradients for rporp
    def _init_rprop_params(self, weights: List[np.ndarray], 
                        biases: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        step_sizes_weights = [0.1 * np.ones_like(w) for w in weights]
        step_sizes_biases = [0.1 * np.ones_like(b) for b in biases]

        # Initialize previous gradients to zero (for weights and biases)
        prev_grad_weights = [np.zeros_like(w) for w in weights]
        prev_grad_biases = [np.zeros_like(b) for b in biases]
        
        return step_sizes_weights, step_sizes_biases, prev_grad_weights, prev_grad_biases

    # Rprop update weights and biases
    def _rprop_update(self, weights: List[np.ndarray], 
                    biases: List[np.ndarray], 
                    grad_weights: List[np.ndarray], 
                    grad_biases: List[np.ndarray], 
                    step_sizes_weights: List[np.ndarray], 
                    step_sizes_biases: List[np.ndarray],
                    prev_grad_weights: List[np.ndarray], 
                    prev_grad_biases: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        eta_plus = 1.2
        eta_minus = 0.5
        max_step = 50.0
        min_step = 1e-6
        
        new_weights = []
        new_biases = []
        
        for i in range(len(weights)):
            # Weight updates
            change_w = np.sign(grad_weights[i] * prev_grad_weights[i])
            step_sizes_weights[i] = np.where(change_w > 0, np.minimum(step_sizes_weights[i] * eta_plus, max_step),
                                            np.where(change_w < 0, np.maximum(step_sizes_weights[i] * eta_minus, min_step),
                                                    step_sizes_weights[i]))
            weight_update = -np.sign(grad_weights[i]) * step_sizes_weights[i]
            weights[i] += weight_update
            prev_grad_weights[i] = np.where(change_w < 0, 0, grad_weights[i])
            
            # Bias updates
            change_b = np.sign(grad_biases[i] * prev_grad_biases[i])
            step_sizes_biases[i] = np.where(change_b > 0, np.minimum(step_sizes_biases[i] * eta_plus, max_step),
                                            np.where(change_b < 0, np.maximum(step_sizes_biases[i] * eta_minus, min_step),
                                                    step_sizes_biases[i]))
            bias_update = -np.sign(grad_biases[i]) * step_sizes_biases[i]
            biases[i] += bias_update
            prev_grad_biases[i] = np.where(change_b < 0, 0, grad_biases[i])
            
            new_weights.append(weights[i])
            new_biases.append(biases[i])
        
        return new_weights, new_biases, prev_grad_weights, prev_grad_biases, step_sizes_weights, step_sizes_biases

    def _update_params(
        self,
        weights: List[np.ndarray], 
        biases: List[np.ndarray], 
        dW: List[np.ndarray], 
        dB: List[np.ndarray], 
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        new_weights = [] * len(weights)
        new_biases = [] * len(biases)

        for i in range(len(weights)):
            new_weights[i] = weights[i] - self._learning_rate * dW[i]
            new_biases[i] = biases[i] - self._learning_rate * dB[i]

        return new_weights, new_biases