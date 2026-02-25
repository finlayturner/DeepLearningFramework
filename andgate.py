from DL import Linear, NN
import time
import numpy as np

start_time = time.time()

n_samples = 1000
input_dim = 5
lr = 0.1
layer = Linear(input_dim, 1, lr=lr, activation='sigmoid')

model = NN(input_dim)
model.add_layer(layer)

X_train = np.random.randint(0, 2, size=(n_samples, input_dim)).astype(float)
y_train = ((X_train[:, 0] == 1) & (X_train[:, 4] == 1)).astype(float)
loss = 0
for epoch in range(50000):
    # Pick a random sample
    i = np.random.randint(0, n_samples)
    x = X_train[i].reshape(1, input_dim)
    target = y_train[i]

    # Forward
    output = model.forward(x)
    loss += 0.5 * (output - target)**2

    if epoch % 5000 == 0:
        loss = loss / epoch
        print(f"Epoch {epoch}, Avg Loss: {loss[0][0]}")
        loss = 0
        
    # Loss
    d_loss = output - target

    # Backprop
    model.backward(d_loss)

    # Gradient Descent
    model.grad_descent()

print("\n--- TEST: Does it ignore noise? ---")
test_clean = np.array([1, 0, 0, 0, 1])
test_noisy = np.array([1, 1, 1, 1, 1])
test_false = np.array([1,1,0,1,0])
test_falsetwo = np.array([0,1,0,1,1])

print(f"Clean Input [1,0,0,0,1] -> Output: {model.forward(test_clean)}")
print(f"Noisy Input [1,1,1,1,1] -> Output: {model.forward(test_noisy)}")
print(f"Noisy Input [1,1,0,1,0] -> Output: {model.forward(test_false)}")
print(f"Noisy Input [0,1,0,1,1] -> Output: {model.forward(test_falsetwo)}")

print("--- %s seconds ---" % (time.time() - start_time))

print(model)