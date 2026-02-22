from DL import Linear, NN
import time
import numpy as np

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