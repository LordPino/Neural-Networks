from typing import  Callable, List
import numpy as np

# One hot encoding
def one_hot(y: np.ndarray) -> np.ndarray:
    print(f"Size: {y.size}, v: {y.max()}")
    one_hot_y = np.zeros((y.size, int(y.max()) + 1))
    y_int = y.astype(int)
    one_hot_y[np.arange(y.size), y_int] = 1
    
    return one_hot_y

# Generic derivative of a function
def function_derivative(
    function: Callable[[np.ndarray], np.ndarray], 
    x: np.ndarray, 
    dx: float = 1e-6
) -> np.ndarray:
    return (function(x + dx) - function(x)) / dx

def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]  # Calculate loss
    return loss

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
def ReLU_function(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

# Derivative of the ReLU function
def ReLU_prime(z: np.ndarray) -> np.ndarray:
    output = np.where(z > 0, 1, 0)
    return output

# Returns the indices of the maximum values along an axis.
def get_predictions(a: np.ndarray) -> np.ndarray:
    return np.argmax(a, axis=1)

# Accuracy function
def get_accuracy(predictions: np.ndarray, y: np.ndarray) -> float:
    y = np.argmax(y, axis=1)
    return np.sum(predictions == y) / y.size

def flatten(input, stride, kernel_size):
    flattened_patches = []

    w = input.shape[0]
    h = input.shape[1]
    for i in range(0, w, stride):
        if i+kernel_size > w:
            continue
        for j in range(0, h, stride):
            if j+kernel_size > h:
                continue
            # Extract the patch
            patch = input[i:i+kernel_size, j:j+kernel_size]
            # Flatten and add the patch to the list
            flattened_patches.append(patch.flatten())
    return flattened_patches

def flatten_list(x_train, stride, kernel_size):
    results = []
    for i in range(0, len(x_train)):
        f = flatten(x_train[i], stride, kernel_size)
        results.append(f)
    
    return results

def reshape_columns_to_square_matrices(A):
    # Check if the number of rows (M) is a perfect square
    M = A.shape[0]
    if np.sqrt(M) % 1 != 0:
        raise ValueError("The number of rows (M) must be a perfect square.")

    # Calculate the side length of the square matrix
    side_length = int(np.sqrt(M))

    # Initialize an empty list to store the square matrices
    square_matrices = []

    # Iterate through each column of A
    for column in range(A.shape[1]):
        # Extract the current column and reshape it into a square matrix
        square_matrix = A[:, column].reshape(side_length, side_length)
        square_matrices.append(square_matrix)

    return square_matrices

def reconstruct_from_flattened_patches(flattened_patches, orig_w, orig_h, stride, kernel_size):
    reconstructed_image = np.zeros((orig_w, orig_h))
    patch_idx = 0
    for i in range(0, orig_w, stride):
        if i + kernel_size > orig_w:
            continue
        for j in range(0, orig_h, stride):
            if j + kernel_size > orig_h:
                continue
            patch = flattened_patches[patch_idx].reshape(kernel_size, kernel_size)
            reconstructed_image[i:i+kernel_size, j:j+kernel_size] = patch
            patch_idx += 1
    return reconstructed_image

def matrix_multiply_and_flatten(M, matrices_list):
    # M is the matrix with dimensions K x P
    # matrices_list is the list of S matrices, each of dimensions P x P
    
    flattened_results = []
    
    # Iterate over each P x P matrix in the list
    for matrix in matrices_list:
        # Perform matrix multiplication M * matrix (resulting in a K x P matrix)
        result = np.dot(M, matrix.flatten())
        
        # Flatten the resulting matrix and append it to the list
        flattened_results.append(result)
    
    # Concatenate all flattened results side by side
    # This results in a matrix of dimensions (K*P) x S
    final_matrix = np.column_stack(flattened_results)
    
    return final_matrix