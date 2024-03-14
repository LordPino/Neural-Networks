import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

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
            patch = input[i:i+kernel_size, j:j+kernel_size]
            flattened_patches.append(patch.flatten())
    return flattened_patches

def flatten_list(x_train):
    results = []
    for i in range(len(x_train)):
        f = flatten(x_train[i], 2, 4)
        results.append(f)
    return results

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

# Create a simple 8x8 grayscale image
orig_image = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 9],
    [3, 4, 5, 6, 7, 8, 9, 10],
    [4, 5, 6, 7, 8, 9, 10, 11],
    [5, 6, 7, 8, 9, 10, 11, 12],
    [6, 7, 8, 9, 10, 11, 12, 13],
    [7, 8, 9, 10, 11, 12, 13, 14],
    [8, 9, 10, 11, 12, 13, 14, 15]
])

# Flatten the image into patches
flattened_patches = flatten(orig_image, 2, 4)

# Assume each "image" in x_train is just our original image for demonstration
x_train = [orig_image]
flattened_list = flatten_list(x_train)

# Flatten the list since our reconstruction function expects a flat list of patches
flattened_patches = [item for sublist in flattened_list for item in sublist]

# Reconstruct the image
reconstructed_image = reconstruct_from_flattened_patches(flattened_patches, 8, 8, 2, 4)

# Print both the original and reconstructed images for comparison
print("Original Image:\n", orig_image)
print("flattened_patches:\n", flattened_patches)
print("Reconstructed Image:\n", reconstructed_image)




import numpy as np

def matrix_multiply_and_flatten(M, matrices_list):
    # M is the matrix with dimensions K x P
    # matrices_list is the list of S matrices, each of dimensions P x P
    
    flattened_results = []
    
    # Iterate over each P x P matrix in the list
    for matrix in matrices_list:
        # Perform matrix multiplication M * matrix (resulting in a K x P matrix)
        result = np.dot(M, matrix)
        
        # Flatten the resulting matrix and append it to the list
        flattened_results.append(result.flatten())
    
    # Concatenate all flattened results side by side
    # This results in a matrix of dimensions (K*P) x S
    final_matrix = np.column_stack(flattened_results)
    
    return final_matrix

# Example usage
K, P, S = 3, 2, 4  # Example dimensions
M = np.random.rand(K, P)  # Example matrix M of dimensions K x P
matrices_list = [np.random.rand(P, P) for _ in range(S)]  # List of S matrices, each of dimensions P x P

result_matrix = matrix_multiply_and_flatten(M, matrices_list)
print("Result Matrix:\n", result_matrix)
