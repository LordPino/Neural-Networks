
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.datasets import mnist
from library.network import train
from library.activations import Sigmoid
from library.convolutional import Convolutional
from library.dense import Dense
from library.reshape import Reshape
from library.utils import derivatie_cross_entropy
from keras.utils import to_categorical

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y, num_classes=10)  # One-hot encode y
    y = y.reshape(len(y), 10, 1)

    # No need to reshape y to (len(y),1, 10, 1) unless specifically required by your network architecture
    return x, y

def test_prediction(index: int, x: np.ndarray, y: np.ndarray):
    current_image = x[:, index, None]
    prediction = network.make_predictions(x=x[:, index, None])
    label = y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

train(network, None, loss_prime=derivatie_cross_entropy, x_train=x_train, y_train=y_train, epochs=20, learning_rate=0.1, use_r_prop=False)

test_prediction(0, x_train, y_train, network=network)
test_prediction(1, x_train, y_train, network=network)
test_prediction(2, x_train, y_train, network=network)
test_prediction(3, x_train, y_train, network=network)