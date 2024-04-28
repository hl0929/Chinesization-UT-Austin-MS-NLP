import numpy as np
import matplotlib.pyplot as plt


# 定义函数 f(x)
def f(x):
    return x ** 2 + 5 * np.sin(x)


# 定义函数 f(x) 的导数
def f_derivative(x):
    return 2 * x + 5 * np.cos(x)


# 梯度下降算法
def gradient_descent(x_start, learning_rate, num_iterations):
    x = x_start
    history = [x]                      # 保存每次迭代后的 x 值
    for _ in range(num_iterations):
        gradient = f_derivative(x)     # 计算梯度
        x -= learning_rate * gradient  # 更新参数 x
        history.append(x)              # 保存更新后的 x 值
    return x, history


# 设置初始参数和学习率
x_start = -3         # 初始参数值
learning_rate = 0.1  # 学习率
num_iterations = 50  # 迭代次数

# 运行梯度下降算法
x_min, x_history = gradient_descent(x_start, learning_rate, num_iterations)

# 输出最小值和最小值对应的 x 值
print("最小值:", f(x_min))
print("最小值对应的 x 值:", x_min)

# 绘制函数曲线和梯度下降过程
x = np.linspace(-5, 5, 100)
y = f(x)
plt.plot(x, y, label='f(x)')
plt.scatter(x_history, f(np.array(x_history)), c='r', label='Gradient Descent Point')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()