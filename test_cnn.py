
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.datasets import mnist
from library.layer import Layer
from library.network import predict, train
from library.activations import ReLU, Sigmoid
from library.convolutional import Convolutional
from library.dense import Dense
from library.reshape import Reshape
from library.utils import derivatie_cross_entropy, get_accuracy, get_predictions
from keras.utils import to_categorical

def preprocess_data(x, y, limit):
    unique_classes = np.unique(y)  # Find the unique classes in y
    all_indices = []  # To store selected indices for all classes

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0][:limit]  # Get indices for class up to limit
        all_indices.append(cls_indices)
    
    all_indices = np.hstack(all_indices)  # Combine indices from all classes
    all_indices = np.random.permutation(all_indices)  # Shuffle the combined indices

    x = x[all_indices]  # Select balanced set of features
    y = y[all_indices]  # Select balanced set of labels

    
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y, num_classes=10)  # One-hot encode y
    y = y.reshape(len(y), 10, 1)

    # No need to reshape y to (len(y),1, 10, 1) unless specifically required by your network architecture
    return x, y

def test_prediction(index: int, x: np.ndarray, y: np.ndarray, network):
    prediction = make_predictions(x=x[index,:, None], network=network)
    label = y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

def make_predictions(x: np.ndarray, network: list[Layer]) -> np.ndarray:
    print(x.shape)
    output = predict(network, x)
    predictions = get_predictions(output)
    
    return predictions

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

network = [
    Convolutional((1, 28, 28), 3, 1),
    ReLU(),
    Reshape((1, 26, 26), (1 * 26 * 26, 1)),
    Dense(1 * 26 * 26, 100),
    ReLU(),
    Dense(100, 10),
    Sigmoid()
]

train(network, None, loss_prime=derivatie_cross_entropy, x_train=x_train, y_train=y_train, epochs=20, learning_rate=0.1, use_r_prop=False)

for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")