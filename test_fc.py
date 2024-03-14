from typing import List
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from library.activations import ReLU, Sigmoid, Softmax
from library.conv_dense import ConvDense
from library.dense import Dense
from library.input_layer import InputLayer

from library.layer import Layer
from library.network import predict, train
from library.reshape import Reshape
from library.utils import cross_entropy_loss, derivatie_cross_entropy, flatten_list, get_accuracy, get_predictions
from keras.utils import to_categorical

#make predictions
def make_predictions(x: np.ndarray, network: List[Layer]) -> np.ndarray:
    
    outputs = []
    for image in x:
        o = predict(network, image)
        outputs.append(o)
    
    outputs = np.array(outputs)
    
    return get_predictions(outputs)
    
def preprocess_data(x: np.ndarray, y: np.ndarray, limit: int = 60000):
    unique_classes = np.unique(y)  # Find the unique classes in y
    all_indices = []  # To store selected indices for all classes

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0][:limit]  # Get indices for class up to limit
        all_indices.append(cls_indices)
    
    all_indices = np.hstack(all_indices)  # Combine indices from all classes
    all_indices = np.random.permutation(all_indices)  # Shuffle the combined indices

    x = x[all_indices]  # Select balanced set of features
    y = y[all_indices]  # Select balanced set of labels
    
    x_train = flatten_list(x, 4, 4)
    x_train = np.array(x_train)
    x_train = x_train.astype("float32") / 255
    #print(x_train.shape) # (60000, 169, 16)
    y_train = to_categorical(y, num_classes=10)  # One-hot encode y
    y_train = y_train.reshape(len(y), 10, 1)
    
    #print(y_train.shape) # (60000, 10, 1)
    return x_train, y_train

network_1K = [
    ConvDense((169, 16), 1), # ConvDense. K = 16x1 
    ReLU(),
    Dense(169, 100),
    Sigmoid(),
    Dense(100, 10),
    Softmax(),
]

network_8K = [
    #InputLayer(2),
    ConvDense((49, 16), 8), 
    ReLU(),
    Reshape((49, 8), (49 * 8, 1)),
    Dense(49 * 8, 49),
    ReLU(),
    Dense(49, 10),
    Softmax(use_cross_entropy=True),
]
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 2500)

'''
errors_1k, accuracies_1K = train(
    x_train=x_train,
    y_train=y_train,
    epochs=40,
    learning_rate=0.01,
    loss_prime=derivatie_cross_entropy,
    loss_function=cross_entropy_loss,
    network=network_1K,
    use_r_prop=True
)

dev_predictions = make_predictions(x_test, network_1K)
print("Test accuracy:")
test_accuracy_network_1K = get_accuracy(dev_predictions, y_test)
print(test_accuracy_network_1K)

'''
errors_8k, accuracies_8K = train(
    x_train=x_train,
    y_train=y_train,
    epochs=8,
    learning_rate=0.01,
    loss_prime=derivatie_cross_entropy,
    loss_function=cross_entropy_loss,
    network=network_8K,
    use_r_prop=False
)

dev_predictions = make_predictions(x_test, network_8K)
print("Test accuracy:")
test_accuracy_network_8K = get_accuracy(dev_predictions, y_test)
print(test_accuracy_network_8K)