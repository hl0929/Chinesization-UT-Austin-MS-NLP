import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义单隐藏层神经网络
class SingleLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SingleLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 参数设置
input_size = 784  # 输入层大小（如28x28的图像展开为784）
hidden_size = 128  # 隐藏层大小
output_size = 10  # 输出层大小（如10个类别）
learning_rate = 0.001
num_epochs = 20
batch_size = 64

# 生成一些随机数据
x_train = torch.randn(600, input_size)  # 600个训练样本
y_train = torch.randint(0, output_size, (600,))  # 600个训练标签
x_test = torch.randn(100, input_size)  # 100个测试样本
y_test = torch.randint(0, output_size, (100,))  # 100个测试标签

# 创建数据集和数据加载器
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 实例化神经网络和损失函数
model = SingleLayerNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练神经网络
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        # 前向传递
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试神经网络
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')