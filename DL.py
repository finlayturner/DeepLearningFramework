import numpy as np
from activations import Activations

class Linear:
    def __init__(self, input_size: int, output_size: int, lr=0.1, activation='relu'):
        self.shape = (output_size, input_size)
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        self.B = np.zeros(output_size).reshape(-1, 1)
        self.grad = 0

        activation_map = {
            'relu': (Activations.relu, Activations.relu_d),
            'sigmoid': (Activations.sigmoid, Activations.sigmoid_d),
            'none': (lambda x: x, lambda x: np.ones_like(x))
        }

        self.activate, self.activate_d = activation_map[activation.lower()]

        # Learning Rate
        self.lr = lr

    def forward(self, X):
        self.X = X.reshape(-1, 1)
        self.Z = self.W @ self.X + self.B
        return np.maximum(0, self.Z) #RELU
    
    def backward(self, incoming_grad):
        # (dl/dy * dy/dz)
        d_relu = (self.Z > 0).astype(float)
        d_relu = incoming_grad * d_relu

        # Bias delta
        self.grad_B = d_relu
        # (dl/dW)
        self.grad_W = d_relu @ self.X.T

        return self.W.T @ d_relu
    
    def grad_descent(self):
        # ???
        self.W -= self.lr * self.grad_W
        self.B -= self.lr * self.grad_B

class NN:
    def __init__(self, input_size: int):
        
        self.input_size = input_size
        self.layers = []
    
    def add_layer(self, layer):
        """
        Append new layer to neural network
        """
        self.layers.append(layer)

    def forward(self, X):
        """
        Forward propagate through network layers
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, x):
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            x = layer.backward(x)
    
    def grad_descent(self):
        for layer in self.layers:
            layer.grad_descent()
    
    def __str__(self):
        count = 1
        for layer in self.layers:
            print(f"Layer {count}")
            print(f"w{layer.W}, B{layer.B}")
            count+=1
        return ""