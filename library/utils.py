from typing import Callable
import numpy as np

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
    # Shift z by subtracting its max value from all elements
    shift_z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

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
    return np.sum(predictions == y) / y.size
