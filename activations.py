import numpy as np

class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_d(z):
        return (z > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_d(z):
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)