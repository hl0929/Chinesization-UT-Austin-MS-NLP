import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(Seq2Seq, self).__init__()
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, src, tgt):
        # 编码器
        _, (hidden, cell) = self.encoder(src)
        
        # 解码器
        outputs, _ = self.decoder(tgt, (hidden, cell))
        
        # 全连接层
        predictions = self.fc(outputs)
        
        return predictions


# 超参数
input_dim = 10   # 输入的特征维度
output_dim = 10  # 输出的特征维度
hidden_dim = 16  # 隐藏层维度
n_layers = 2     # LSTM层数
seq_len = 5      # 序列长度
batch_size = 2   # 批次大小

# 创建示例数据（随机生成）
np.random.seed(0)
torch.manual_seed(0)

src_data = torch.randn(batch_size, seq_len, input_dim)  # 输入序列
tgt_data = torch.randn(batch_size, seq_len, output_dim)  # 目标序列

# 打印数据形状
print("Source data shape:", src_data.shape)
print("Target data shape:", tgt_data.shape)

    
# 初始化模型、损失函数和优化器
model = Seq2Seq(input_dim, output_dim, hidden_dim, n_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    
    output = model(src_data, tgt_data)
    loss = criterion(output, tgt_data)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')