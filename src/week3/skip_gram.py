import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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


# 定义Skip-gram模型
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.fc(embedded)
        return output


# 定义数据集
class SkipGramDataset(Dataset):
    def __init__(self, corpus, word2idx, window_size):
        self.data = []
        for sentence in corpus:
            tokens = sentence.lower().split()
            for i in range(window_size, len(tokens) - window_size):
                center_word = word2idx[tokens[i]]
                context_words = [word2idx[tokens[j]] for j in range(i - window_size, i + window_size + 1) if j != i]
                for context_word in context_words:
                    self.data.append((center_word, context_word))

    def __getitem__(self, index):
        center_word, context_word = self.data[index]
        return torch.tensor(center_word), torch.tensor(context_word)

    def __len__(self):
        return len(self.data)


# 训练参数设置
embedding_dim = 10
window_size = 2
batch_size = 1
lr = 0.001
epochs = 100

# 创建数据加载器
dataset = SkipGramDataset(corpus, word2idx, window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建模型和优化器
model = SkipGram(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练过程
for epoch in range(epochs):
    total_loss = 0.0
    for center_word, context_word in dataloader:
        optimizer.zero_grad()
        output = model(center_word)
        loss = criterion(output, context_word)
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