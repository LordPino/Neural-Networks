import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from enum import Enum

class FunctionError(Enum):
    MSE = 1
    CROSS_ENTROPY = 2

def init_params(num_layers: int, input_val: int, output_val: int):
    if num_layers < 2:
        raise ValueError("Number of layers must be at least 2")
    
    # Preallocation weights and biases
    weights = [None] * (num_layers - 1)
    biases = [None] * (num_layers - 1)

    # First layer 
    weights[0] = np.random.rand(output_val, input_val)  - 0.5
    biases[0] = np.random.rand(output_val, 1) - 0.5

    # Middle layers and output layer
    for i in range(1, num_layers - 1):
        weights[i] = np.random.rand(output_val, output_val) - 0.5
        biases[i] = np.random.rand(output_val, 1) - 0.5

    return weights, biases

def forward_prop(x, weights, biases, activation_functions, output_function):
    num_layers = len(weights)
    
    # Preallocate lists for z and a with the appropriate sizes
    z = [None] * num_layers
    a = [None] * (num_layers + 1)  # +1 for the input layer

    # Set the input layer
    a[0] = x

    # Iterate through each layer except the last one
    for i in range(num_layers - 1):
        z[i] = np.dot(weights[i], a[i]) + biases[i]
        a[i + 1] = activation_functions[i](z[i])

    # Handle the output layer separately
    z[-1] = np.dot(weights[-1], a[-2]) + biases[-1]
    a[-1] = output_function(z[-1])

    return a, z

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    
    return one_hot_y

def function_derivative(function, x, dx=1e-6):
    return (function(x + dx) - function(x)) / dx

def derivatie_cross_entropy(y_true, y_pred):
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Aggiunto 1e-9 per la stabilità numerica
    return loss

def derivative_mse(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def back_prop(x, y, z, a, weights, biases, activation_functions, activation_derivatives, function_error: FunctionError, use_softmax: bool):
    m = y.shape[0]
    one_hot_y = one_hot(y)

    # Initialize gradients
    dW = [None] * len(weights)
    dB = [None] * len(biases)
    dZ = [None] * len(z)

    # Output layer error
    if function_error == FunctionError.MSE:
        dZ[-1] = derivative_mse(one_hot_y, a[-1])
    elif function_error == FunctionError.CROSS_ENTROPY:
        if use_softmax:
            dZ[-1] = a[-1] - one_hot_y
        else:
            dZ[-1] = derivatie_cross_entropy(one_hot_y, a[-1])

    # Output layer gradients
    dW[-1] = (1/m) * np.dot(dZ[-1], a[-2].T)
    dB[-1] = (1/m) * np.sum(dZ[-1])

    # Backpropagation through layers
    for i in range(len(weights) - 2, -1, -1):
        dA = np.dot(weights[i + 1].T, dZ[i + 1])
        if activation_derivatives[i] is not None:
            dZ[i] = dA * activation_derivatives[i](z[i])
        else:
            dZ[i] = dA * function_derivative(activation_functions[i], z[i])
        dW[i] = (1/m) * np.dot(dZ[i], a[i].T)
        dB[i] = (1/m) * np.sum(dZ[i])

    return dW, dB

def update_params(weights, biases, dW, dB, learning_rate):
    new_weights = [None] * len(weights)
    new_biases = [None] * len(biases)

    for i in range(len(weights)):
        new_weights[i] = weights[i] - learning_rate * dW[len(weights) - 1 - i]
        new_biases[i] = biases[i] - learning_rate * dB[len(biases) - 1 - i]

    return new_weights, new_biases

def get_predictions(a):
    return np.argmax(a, axis=0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradint_descent(X, Y, epochs, learning_rate, num_layers, input_val, output_val, activation_functions, activation_derivatives, output_function, function_error: FunctionError, use_softmax: bool):
    weights, biases = init_params(num_layers, input_val, output_val)

    for i in range(epochs):
        a, z = forward_prop(X, weights, biases, activation_functions, output_function)
        dW, dB = back_prop(X, Y, z, a, weights, biases, activation_functions, activation_derivatives, function_error, use_softmax)
        weights, biases = update_params(weights, biases, dW, dB, learning_rate)
        if i % 10 == 0:
            print("epoch: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(a[-1]), Y))

    return weights, biases

def soft_max(z):
    return np.exp(z) / sum(np.exp(z))

def soft_max_derivative(z):
    return soft_max(z) * (1 - soft_max(z))

def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    return np.where(z > 0, 1, 0)

data = pd.read_csv(r'.\test_data\train.csv\train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0: 1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1: n]

data_train = data[1000: m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

W, B = gradint_descent(X=X_train, Y=Y_train, epochs=500, learning_rate=0.1, num_layers=2, input_val=784, output_val=10, activation_functions=[ReLU], activation_derivatives=[ReLU_derivative], output_function=soft_max, function_error=FunctionError.CROSS_ENTROPY, use_softmax=True)