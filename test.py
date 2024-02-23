
from typing import List
import numpy as np
import pandas as pd
from library.layer import Layer, OutputLayer
from library.network import FunctionError, Network
from library.utils import ReLU, ReLU_derivative, get_accuracy, soft_max, soft_max_derivative
from matplotlib import pyplot as plt

def test_prediction(
                    index: int,
                    x: np.ndarray, 
                    y: np.ndarray, 
                    network: Network):
    current_image = x[:, index, None]
    prediction = network.make_predictions(x=x[:, index, None])
    label = y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

data = pd.read_csv(r'.\\train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0: 1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1: n]
X_dev = X_dev / 255.

data_train = data[1000: m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.

network = Network()
network.x(X_train)
network.y(Y_train)
network.epochs(500)
network.learning_rate(0.1)
network.error_function(FunctionError.CROSS_ENTROPY)
network.use_rprop(True)
network.use_softmax(True)
network.add_layer(Layer(784, activation_function=ReLU, activation_derivate=ReLU_derivative))
network.add_layer(OutputLayer(10, output_function=soft_max, output_derivate=soft_max_derivative))

network.train()

test_prediction(0, X_train, Y_train, network=network)
test_prediction(1, X_train, Y_train, network=network)
test_prediction(2, X_train, Y_train, network=network)
test_prediction(3, X_train, Y_train, network=network)

dev_predictions = network.make_predictions(X_dev)
print(get_accuracy(dev_predictions, Y_dev))