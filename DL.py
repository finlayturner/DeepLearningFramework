import numpy as np

class Linear:
    def __init__(self, input_size: int, output_size: int, lr=0.1):
        self.shape = (input_size, output_size)
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.B = np.zeros(output_size)

        # Learning Rate
        self.lr = lr

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.B
        return np.maximum(0, self.Z)

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
    
    def __str__(self):
        count = 1
        for layer in self.layers:
            print(f"Layer {count}")
            print(f"w{layer.W}, B{layer.B}")
            count+=1
        return ""