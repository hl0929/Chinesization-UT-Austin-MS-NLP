import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR input and output
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])  # XOR outputs

# Seed for reproducibility
np.random.seed(42)

# Initialize weights randomly with mean 0
input_layer_neurons = X.shape[1]
hidden_layer_neurons = 2  # You can change this number
output_neurons = 1

# Weight matrix for input to hidden layer
W1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))

# Weight matrix for hidden to output layer
W2 = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

# Bias for hidden and output layers
b1 = np.random.uniform(size=(1, hidden_layer_neurons))
b2 = np.random.uniform(size=(1, output_neurons))

# Learning rate
lr = 0.1

# Training the MLP
epochs = 10000
for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    
    # Backpropagation
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)
    
    error_hidden_layer = d_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)
    
    # Updating the weights and biases
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden_layer) * lr
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

# Testing the MLP
print("Predictions after training:")
for x in X:
    hidden_input = np.dot(x, W1) + b1
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    
    print(f"{x} -> {final_output.round()}")