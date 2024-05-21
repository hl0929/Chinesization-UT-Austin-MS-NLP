import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax

# 数据预处理
corpus = [
    "I enjoy playing football, do you know",
    "I like watching movies, do you like",
    "I love eating pizza, are you?"
]

# 构建词汇表
word2idx = {}
idx2word = {}
idx = 0
for sentence in corpus:
    for word in sentence.lower().split():
        if word not in word2idx:
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1

vocab_size = len(word2idx)

# 定义FastText模型
class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)  # 池化操作改为按行平均
        output = self.fc(pooled)
        return output

# 定义数据集
class FastTextDataset(Dataset):
    def __init__(self, corpus, word2idx):
        self.data = []
        for sentence in corpus:
            tokens = sentence.lower().split()
            label = word2idx[tokens[0]]
            text = [word2idx[word] for word in tokens[1:]]
            self.data.append((text, label))

    def __getitem__(self, index):
        text, label = self.data[index]
        return torch.tensor(text), torch.tensor(label)

    def __len__(self):
        return len(self.data)

# 训练参数设置
embedding_dim = 10
batch_size = 1
lr = 0.001
epochs = 100

# 创建数据加载器
dataset = FastTextDataset(corpus, word2idx)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型和优化器
model = FastText(vocab_size, embedding_dim, len(idx2word))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练过程
for epoch in range(epochs):
    total_loss = 0.0
    for text, label in dataloader:
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# 获取训练后的词向量
word_vectors = model.embedding.weight.data

# 打印词向量
for i in range(vocab_size):
    word = idx2word[i]
    vector = word_vectors[i]
    print(f"{word}: {vector}")