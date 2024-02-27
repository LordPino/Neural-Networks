
from enum import Enum
from library.layer import Layer
from library.utils import get_accuracy, get_predictions
import numpy as np

def predict(network: list[Layer], input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network: list[Layer], loss_function, loss_prime, x_train, y_train: np.ndarray, epochs: int, learning_rate: float = 0.01, use_r_prop = False):
    for e in range(epochs):
        error = 0
        outputs = []
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)
            outputs.append(output)
            error += loss_function(y, output)
            
            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate, use_r_prop)
            
        outputs = np.array(outputs)
        
        error /= len(x_train)
        print(f"{e + 1}/{epochs},\nerror={error}")
        print("Accuracy: ", get_accuracy(get_predictions(outputs), y_train))
