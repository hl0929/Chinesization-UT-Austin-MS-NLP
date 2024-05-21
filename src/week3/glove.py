import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 假设数据集
# 词汇表
vocab = ['the', 'cat', 'dog', 'in', 'hat']
vocab_size = len(vocab)

# 构建一个非常简单的共现矩阵（通常这个矩阵是从大量文本中统计得到）
# 这里使用随机值填充，实际应用中应基于真实文本统计
X = np.random.randint(1, 10, (vocab_size, vocab_size))

# 将numpy数组转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float)

# 定义GloVe模型
class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVe, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_dim)
        self.U = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, i, j):
        w_i = self.W(i)
        u_j = self.U(j)
        dot_product = torch.sum(w_i * u_j, dim=1)
        return dot_product

# 初始化模型和优化器
embedding_dim = 5  # 选择一个较小的维度以便快速演示
glove_model = GloVe(vocab_size, embedding_dim)
optimizer = optim.Adam(glove_model.parameters(), lr=0.05)

# 训练函数
def train_glove():
    for epoch in range(10):  # 进行少量迭代以示例
        for i in range(vocab_size):
            for j in range(vocab_size):
                optimizer.zero_grad()
                
                # 转换索引为张量
                i_idx = torch.tensor([i], dtype=torch.long)
                j_idx = torch.tensor([j], dtype=torch.long)
                
                # 计算目标值（这里简化处理，实际GloVe使用的是更复杂的权重函数）
                target = torch.log(X_tensor[i][j])
                
                # 计算模型输出
                pred = glove_model(i_idx, j_idx)
                
                # 计算损失（这里仅作为示例，实际GloVe损失函数更复杂）
                loss = (pred - target) ** 2
                
                # 反向传播与优化
                loss.backward()
                optimizer.step()

# 训练模型
train_glove()

# 查看学习到的词嵌入并计算词之间的余弦相似度
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
word1 = 'the'
word2 = 'cat'

word1_idx = word_to_idx[word1]
word2_idx = word_to_idx[word2]

w1 = glove_model.W(torch.tensor([word1_idx]))
w2 = glove_model.W(torch.tensor([word2_idx]))

similarity = cosine_similarity(w1.detach().numpy(), w2.detach().numpy())
print(f"Cosine similarity between '{word1}' and '{word2}': {similarity[0][0]}")