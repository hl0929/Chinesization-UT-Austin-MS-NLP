import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.lr = lr
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(x)
        return self.activation_fn(z)

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1) # Insert bias term in x
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x

# Input data and corresponding labels
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

d = np.array([0, 1, 1, 0])  # XOR labels

# Create perceptron instance
perceptron = Perceptron(input_size=2)

# Train the perceptron model
perceptron.fit(X, d)

# Test the perceptron model
print("Predictions:")
for x in X:
    x = np.insert(x, 0, 1)  # Insert bias term
    print(f"{x[1:]} -> {perceptron.predict(x)}")