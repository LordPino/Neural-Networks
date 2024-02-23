
from typing import List
import numpy as np
import pandas as pd
from library.layer import Layer, OutputLayer
from library.network import FunctionError, Network
from library.train_result import TrainResult
from library.types import ActivationFunction, OutputFunction
from library.utils import ReLU, ReLU_derivative, soft_max, soft_max_derivative

def test_prediction(self,
                        index: int,
                        x: np.ndarray, 
                        y: np.ndarray,
                        ):
        range = x[:, index, None]
        current_image = range
        prediction = self.make_predictions(x=range)
        label = y[index]

        print("Prediction: ", prediction)
        print("Label: ", label)
        
        current_image = current_image.reshape((28, 28)) * 255
        
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

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

network = Network(epochs=500)
network.learning_rate(0.1)
network.error_function(FunctionError.CROSS_ENTROPY)
network.use_rprop(True)
network.use_softmax(True)
network.add_layer(Layer(784, activation_function=ReLU, activation_derivate=ReLU_derivative))
network.add_layer(OutputLayer(10, output_function=soft_max, output_derivate=soft_max_derivative))

train_result = network.train(X=X_train, Y=Y_train)

