import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(X, y, theta):
    y_pred = sigmoid(np.dot(X, theta))
    epsilon = 1e-5  # 用于避免log(0)的情况
    loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
    return loss


def gradient_descent(X, y, learning_rate, num_iterations):
    num_samples, num_features = X.shape
    theta = np.zeros((num_features, 1))
    loss_history = []

    for _ in range(num_iterations):
        y_pred = sigmoid(np.dot(X, theta))
        loss = compute_loss(X, y, theta)
        gradient = (1/num_samples) * np.dot(X.T, (y_pred - y))
        theta -= learning_rate * gradient
        loss_history.append(loss)

    return theta, loss_history


np.random.seed(0)
num_samples = 100
X_positive = np.random.normal(loc=2, scale=1, size=(num_samples, 2))
X_negative = np.random.normal(loc=-2, scale=1, size=(num_samples, 2))
X = np.vstack((X_positive, X_negative))
y = np.vstack((np.ones((num_samples, 1)), np.zeros((num_samples, 1))))

X = np.hstack((np.ones((2*num_samples, 1)), X))
X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)

learning_rate = 0.1
num_iterations = 1000
theta, loss_history = gradient_descent(X, y, learning_rate, num_iterations)

plt.plot(loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Convergence')
plt.show()

# 生成测试样本
X_test = np.array([[1, 1, 1.5], [1, -1, -1.5]])
# 使用训练后的参数进行预测
y_pred = sigmoid(np.dot(X_test, theta))
print(y_pred)