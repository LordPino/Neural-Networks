import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#vedi come chiamare x e y
def init_params(x: int, y: int, weights, bayes):
    for i in range(len(weights)):
        weights[i] = np.random.rand(x, y) - 0.5
    
    for i in range(len(bayes)):
        bayes[i] = np.random.rand(x, 1) - 0.5
    
    return weights, bayes

def forward_prop(x, weights, bayes, activation_function, output_function):
    z = []
    a = [x]
    for i in range(len(weights) - 1):
        z.append(np.dot(weights[i], a[i]) + bayes[i])
        a.append(activation_function[i](z[i]))
        
    z.append(np.dot(weights[-1], a[-1]) + bayes[-1])
    a.append(output_function(z[-1]))
    
    return a, z

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    
    return one_hot_y

def back_prop(x, y, weights, bayes, activation_function, output_function, learning_rate):
    one_hot_y = one_hot(y)
    a, z = forward_prop(x, weights, bayes, activation_function, output_function)
    delta = [None] * len(weights)
    delta[-1] = (a[-1] - one_hot_y) * a[-1] * (1 - a[-1])
    
    for i in range(len(weights) - 2, -1, -1):
        delta[i] = np.dot(weights[i + 1].T, delta[i + 1]) * a[i] * (1 - a[i])
        
    for i in range(len(weights)):
        weights[i] -= learning_rate * np.dot(delta[i], a[i].T)
        bayes[i] -= learning_rate * delta[i]
        
    return weights, bayes

def update_params(x, y, weights, bayes, activation_function, output_function, learning_rate):
    weights, bayes = back_prop(x, y, weights, bayes, activation_function, output_function, learning_rate)
    return weights, bayes

def soft_max(z):
    return np.exp(z) / np.sum(np.exp(z))

def ReLU(z):
    return np.maximum(0, z)