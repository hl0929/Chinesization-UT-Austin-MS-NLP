import numpy as np


class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def predict(self, inputs):
        activation = np.dot(inputs, self.weights) + self.bias
        return np.where(activation >= 0, 1, -1)

    def train(self, inputs, labels, learning_rate, epochs):
        for _ in range(epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                if prediction != labels[i]:
                    update = learning_rate * (labels[i] - prediction)
                    self.weights += update * inputs[i]
                    self.bias += update

# 训练数据
inputs = np.array([[2, 1], [3, 4], [-2, -1], [-3, -2]])
labels = np.array([1, 1, -1, -1])

# 创建感知机对象
perceptron = Perceptron(input_size=2)

# 训练感知机
perceptron.train(inputs, labels, learning_rate=0.1, epochs=10)

# 测试感知机
test_inputs = np.array([[400, 194], [-13, -34]])
predictions = perceptron.predict(test_inputs)

# 输出预测结果
for i in range(len(test_inputs)):
    print(f"Input: {test_inputs[i]}, Prediction: {predictions[i]}")