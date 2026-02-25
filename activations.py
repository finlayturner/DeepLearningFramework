import numpy as np

class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_d(z):
        return (z > 0).astype(float)
    
    @staticmethod
    def relu_w(input_size, output_size):
        return np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_d(z):
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)
    
    @staticmethod
    def sigmoid_w(input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (output_size, input_size))