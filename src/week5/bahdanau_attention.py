import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        encoder_outputs = self.W(encoder_outputs)  # (batch_size, seq_len, hidden_size)

        attention_scores = self.v(torch.tanh(encoder_outputs + self.U(decoder_hidden)))  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)

        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)  # (batch_size, hidden_size)

        return context_vector, attention_weights
    
    
# 创建注意力对象
attention = BahdanauAttention(hidden_size=256)

# 模拟输入数据
encoder_outputs = torch.randn(32, 10, 256)  # (batch_size, seq_len, hidden_size)
decoder_hidden = torch.randn(32, 256)  # (batch_size, hidden_size)

# 前向传播
context_vector, attention_weights = attention(encoder_outputs, decoder_hidden)

# 打印输出结果
print("Context vector shape:", context_vector.shape)  # (32, 256)
print("Attention weights shape:", attention_weights.shape)  # (32, 10, 1)