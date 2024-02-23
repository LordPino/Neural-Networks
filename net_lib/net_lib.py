import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from enum import Enum
from typing import List, Callable, Tuple

# Enumeration for different loss functions
class FunctionError(Enum):
    MSE = 1 # Mean Squared Error
    CROSS_ENTROPY = 2 # Cross Entropy Loss

############################################
# Utility functions
# One hot encoding
def one_hot(y: np.ndarray) -> np.ndarray:
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    
    return one_hot_y

# Generic derivative of a function
def function_derivative(
    function: Callable[[np.ndarray], np.ndarray], 
    x: np.ndarray, 
    dx: float = 1e-6
) -> np.ndarray:
    return (function(x + dx) - function(x)) / dx

# Derivative of the cross entropy loss function
def derivatie_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

# Derivative of the mean squared error loss function
def derivative_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size

# Softmax function
def soft_max(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / sum(np.exp(z))

# Derivative of the softmax function
def soft_max_derivative(z: np.ndarray) -> np.ndarray:
    return soft_max(z) * (1 - soft_max(z))

# ReLU function
def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

# Derivative of the ReLU function
def ReLU_derivative(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0)

# Returns the indices of the maximum values along an axis.
def get_predictions(a: np.ndarray) -> np.ndarray:
    return np.argmax(a, axis=0)

# Accuracy function
def get_accuracy(predictions: np.ndarray, y: np.ndarray) -> float:
    print(predictions, y)
    return np.sum(predictions == y) / y.size


############################################
# Neural Network functions
# Initialize weights and biases
def init_params(neurons_per_layer: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if len(neurons_per_layer) < 2:
        raise ValueError("Number of layers must be at least 2")
    
    # Preallocation weights and biases
    weights = [None] * (len(neurons_per_layer) - 1)
    biases = [None] * (len(neurons_per_layer) - 1)

    # First layer 
    weights[0] = np.random.rand(neurons_per_layer[1], neurons_per_layer[0])  - 0.5
    biases[0] = np.random.rand(neurons_per_layer[1], 1) - 0.5

    # Middle layers and output layer
    for i in range(1, len(neurons_per_layer) - 1):
        weights[i] = np.random.rand(neurons_per_layer[i], neurons_per_layer[i - 1]) - 0.5
        biases[i] = np.random.rand(neurons_per_layer[i], 1) - 0.5
    
    # Output layer
    weights[-1] = np.random.rand(neurons_per_layer[-1], neurons_per_layer[-2]) - 0.5
    biases[-1] = np.random.rand(neurons_per_layer[-1], 1) - 0.5

    return weights, biases

def make_predictions(x: np.ndarray, 
                        weights: List[np.ndarray], 
                        biases: List[np.ndarray], 
                        activation_functions: List[Callable[[np.ndarray], np.ndarray]], 
                        output_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    a, _ = forward_prop(x=x, 
                        weights=weights, 
                        biases=biases, activation_functions=activation_functions, 
                        output_function=output_function)
    predictions = get_predictions(a[-1])
    
    return predictions

def test_prediction(index: int,
                    x: np.ndarray, 
                    y: np.ndarray, 
                    weights: List[np.ndarray], 
                    biases: List[np.ndarray], 
                    activation_functions: List[Callable[[np.ndarray], np.ndarray]], 
                    output_function: Callable[[np.ndarray], np.ndarray]):
    current_image = x[:, index, None]
    prediction = make_predictions (x=x[:, index, None], 
                                    weights=weights, 
                                    biases=biases, 
                                    activation_functions=activation_functions, 
                                    output_function=output_function)
    label = y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
        
# Forward propagation
def forward_prop(
    x: np.ndarray, 
    weights: List[np.ndarray], 
    biases: List[np.ndarray], 
    activation_functions: List[Callable[[np.ndarray], np.ndarray]], 
    output_function: Callable[[np.ndarray], np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray]]: 
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

# Back propagation
def back_prop(
    y: np.ndarray, 
    z: List[np.ndarray], 
    a: List[np.ndarray], 
    weights: List[np.ndarray], 
    biases: List[np.ndarray], 
    activation_functions: List[Callable[[np.ndarray], np.ndarray]],
    activation_derivatives: List[Callable[[np.ndarray], np.ndarray]],
    function_error: FunctionError,
    output_function: Callable[[np.ndarray], np.ndarray],
    output_derivative: Callable[[np.ndarray], np.ndarray],
    use_softmax: bool
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    m = y.shape[0]
    one_hot_y = one_hot(y)

    # Initialize gradients
    dW = [None] * len(weights)
    dB = [None] * len(biases)
    
    # Output layer error
    if function_error == FunctionError.MSE:
        dZ = derivative_mse(one_hot_y, a[-1])
    elif function_error == FunctionError.CROSS_ENTROPY:
        if use_softmax:
            dZ = a[-1] - one_hot_y
        else:
            dZ = derivatie_cross_entropy(one_hot_y, a[-1])

    if function_error == FunctionError.MSE or (function_error == FunctionError.CROSS_ENTROPY and use_softmax is False):
        if output_derivative is not None:
            dZ = dZ * output_derivative(a[-1])
        else:
            dZ = dZ * function_derivative(output_function, a[-1])

    # Output layer gradients
    dW[-1] = (1/m) * np.dot(dZ, a[-2].T)
    dB[-1] = (1/m) * np.sum(dZ)

    # Backpropagation through layers
    for i in range(len(weights) - 2, -1, -1):
        dA = np.dot(weights[i + 1].T, dZ)
        if activation_derivatives[i] is not None:
            dZ = dA * activation_derivatives[i](z[i])
        else:
            dZ = dA * function_derivative(activation_functions[i], z[i])
        dW[i] = (1/m) * np.dot(dZ, a[i].T)
        dB[i] = (1/m) * np.sum(dZ)

    return dW, dB

# Update weights and biases
def update_params(
    weights: List[np.ndarray], 
    biases: List[np.ndarray], 
    dW: List[np.ndarray], 
    dB: List[np.ndarray], 
    learning_rate: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    new_weights = [None] * len(weights)
    new_biases = [None] * len(biases)

    for i in range(len(weights)):
        new_weights[i] = weights[i] - learning_rate * dW[i]
        new_biases[i] = biases[i] - learning_rate * dB[len(biases) - 1 - i]

    return new_weights, new_biases

# Gradient descent
def gradint_descent(
    X: np.ndarray, 
    Y: np.ndarray, 
    epochs: int, 
    learning_rate: float, 
    neurons_per_layer: List[int],
    activation_functions: List[Callable[[np.ndarray], np.ndarray]],
    activation_derivatives: List[Callable[[np.ndarray], np.ndarray]],
    output_function: Callable[[np.ndarray], np.ndarray],
    output_derivative: Callable[[np.ndarray], np.ndarray],
    function_error: FunctionError, 
    use_softmax: bool
)-> Tuple[List[np.ndarray], List[np.ndarray]]:
    weights, biases = init_params(neurons_per_layer)

    for i in range(epochs):
        a, z = forward_prop(x=X, 
                            weights=weights, 
                            biases=biases, 
                            activation_functions=activation_functions, 
                            output_function=output_function)
        dW, dB = back_prop(y=Y, 
                           z=z, 
                           a=a, 
                           weights=weights, 
                           biases=biases, 
                           activation_functions=activation_functions, 
                           activation_derivatives=activation_derivatives, 
                           function_error=function_error, 
                           output_function=output_function,
                           output_derivative=output_derivative,
                           use_softmax=use_softmax)
        weights, biases = update_params(
            weights=weights, 
            biases=biases, 
            dW=dW, 
            dB=dB, 
            learning_rate=learning_rate)
        if i % 10 == 0:
            print("epoch: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(a[-1]), Y))

    return weights, biases

data = pd.read_csv(r'.\test_data\train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0: 1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1: n]
X_dev = X_dev / 255.

data_train = data[1000: m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

neurons_per_layer = [784, 10]
activation_functions = [ReLU]
activation_derivatives = [ReLU_derivative]
output_function = soft_max
output_derivative = soft_max_derivative
learning_rate = 0.1
epochs = 500

W, B = gradint_descent(X=X_train, # Network.train
                       Y=Y_train, # Network.train
                       epochs=epochs, # Network.train
                       learning_rate=learning_rate,  # Network 
                       neurons_per_layer=neurons_per_layer, # Layer
                       activation_functions=activation_functions, # Input & Middle Layer
                       activation_derivatives=activation_derivatives, # Input & Middle Layer
                       output_function=output_function, # Output Layer 
                       output_derivative=output_derivative, # Output Layer
                       function_error=FunctionError.CROSS_ENTROPY, 
                       use_softmax=True)

'''
# Test the predictions

test_prediction(0, X_train, Y_train, W, B, activation_functions, output_function)
test_prediction(1, X_train, Y_train, W, B, activation_functions, output_function)
test_prediction(2, X_train, Y_train, W, B, activation_functions, output_function)
test_prediction(3, X_train, Y_train, W, B, activation_functions, output_function)

dev_predictions = make_predictions(X_dev, W, B, activation_functions, output_function)
print(get_accuracy(dev_predictions, Y_dev))
'''