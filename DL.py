from typing import List
import numpy as np
import time

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


if __name__ == "__main__":
    start_time = time.time()

    n_samples = 1000
    input_dim = 5
    lr = 0.01
    layer = Linear(input_dim, 1, lr=lr)

    X_train = np.random.randint(0, 2, size=(n_samples, input_dim)).astype(float)
    y_train = ((X_train[:, 0] == 1) & (X_train[:, 4] == 1)).astype(float)

    for epoch in range(50000):
        # Pick a random sample
        i = np.random.randint(0, n_samples)
        x = X_train[i].reshape(1, input_dim)
        target = y_train[i]

        # Forward
        output = layer.forward(x)
        
        # Loss
        error = output - target
        
        # Backprop
        layer.W -= lr * (x.T @ error)
        layer.B -= lr * error.flatten()

    print("--- TRAINED WEIGHTS ---")
    for i, weight in enumerate(layer.W.flatten()):
        print(f"Weight for X{i}: {weight:.4f}")
    print("\n--- TEST: Does it ignore noise? ---")
    test_clean = np.array([1, 0, 0, 0, 1])
    test_noisy = np.array([1, 1, 1, 1, 1])
    test_false = np.array([1,1,0,1,0])

    print(f"Clean Input [1,0,0,0,1] -> Output: {layer.forward(test_clean)}")
    print(f"Noisy Input [1,1,1,1,1] -> Output: {layer.forward(test_noisy)}")
    print(f"Noisy Input [1,1,0,1,0] -> Output: {layer.forward(test_false)}")


    print("--- %s seconds ---" % (time.time() - start_time))
