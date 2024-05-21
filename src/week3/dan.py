import torch
import itertools
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import pretrained_aliases
import nltk
from sklearn.metrics import accuracy_score
import torch.optim as optim
from torch.nn import functional as F
import torch.nn as nn
from tqdm import tqdm

# 加载SST-2数据集
dataset = load_dataset("glue", "sst2")

# 分割数据集
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# 加载GloVe预训练词嵌入
glove = pretrained_aliases['glove.6B.300d'](cache='./.vector_cache')

# 数据预处理函数
def preprocess_function(examples):
    sentences = examples['sentence']
    labels = examples['label']
    tokenized_sentences = [nltk.word_tokenize(s.lower()) for s in sentences]  # 分词并转小写
    return {"sentences": tokenized_sentences, "labels": labels}

# 应用预处理
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 定义PyTorch数据集类
class SST2PyTorchDataset(Dataset):
    def __init__(self, data, word_embeddings, seq_length=50):
        self.data = data
        self.word_embeddings = word_embeddings
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data["sentences"])
    
    def __getitem__(self, idx):
        sentence = self.data["sentences"][idx][:self.seq_length]
        label = self.data["labels"][idx]
        vecs = [self.word_embeddings[word] for word in sentence if word in self.word_embeddings.stoi]
        avg_vec = torch.mean(torch.stack(vecs), dim=0) if vecs else torch.zeros(self.word_embeddings.dim)
        return avg_vec, label

# 创建数据集实例
train_pt_dataset = SST2PyTorchDataset(train_dataset, glove)
val_pt_dataset = SST2PyTorchDataset(val_dataset, glove)

# 数据加载器
train_dataloader = DataLoader(train_pt_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_pt_dataset, batch_size=32)

# 定义Deep Averaging Network模型
class DAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 实例化模型和相关组件
model = DAN(input_dim=glove.dim, hidden_dim=100, output_dim=2)
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练函数
def train_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 验证函数
def evaluate(model, dataloader):
    model.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == targets).item()
            total += len(targets)
    return corrects / total

# 训练和验证循环
num_epochs = 1
for epoch in range(num_epochs):
    train_dataloader = list(itertools.islice(train_dataloader, 3))  # 资源足够可以注释掉这行代码
    train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer)
    val_acc = evaluate(model, val_dataloader)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Acc={val_acc*100:.2f}%")

print("Training complete.")
