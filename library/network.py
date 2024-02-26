
from enum import Enum
from library.layer import Layer
from library.utils import get_accuracy, get_predictions

def predict(network: list[Layer], input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network: list[Layer], loss_function, loss_prime, x_train, y_train, epochs: int, learning_rate: float = 0.01, use_r_prop = False):
    for e in range(epochs):
        # error = 0
        o = None
        ly = None
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)
            
            # error += loss_function(y, output)
            
            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate, use_r_prop)
            
            o = output
            ly = y  
        if e % 10 == 0:
            print("epoch: ", e)
            print("Accuracy: ", get_accuracy(get_predictions(o), ly).real)
