import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    def __init__(self, hidden_size, attention_type='dot'):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type

        if attention_type == 'dot':
            self.dot = True
        elif attention_type == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
            self.dot = False
        elif attention_type == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)
            self.dot = False
        else:
            raise ValueError("Invalid attention type.")

    def forward(self, encoder_outputs, decoder_hidden):
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)

        if self.attention_type == 'dot':
            attention_scores = torch.bmm(encoder_outputs, decoder_hidden.transpose(1, 2))  # (batch_size, seq_len, 1)
        elif self.attention_type == 'general':
            attention_scores = torch.bmm(self.W(encoder_outputs), decoder_hidden.transpose(1, 2))  # (batch_size, seq_len, 1)
        elif self.attention_type == 'concat':
            decoder_hidden = decoder_hidden.expand(-1, encoder_outputs.size(1), -1)  # (batch_size, seq_len, hidden_size)
            concat_inputs = torch.cat((encoder_outputs, decoder_hidden), dim=2)  # (batch_size, seq_len, hidden_size * 2)
            attention_scores = self.v(torch.tanh(self.W(concat_inputs)))  # (batch_size, seq_len, 1)
        else:
            raise ValueError("Invalid attention type.")

        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)

        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)  # (batch_size, hidden_size)

        return context_vector, attention_weights
    

# 创建注意力对象
attention = LuongAttention(hidden_size=256, attention_type='dot')

# 模拟输入数据
encoder_outputs = torch.randn(32, 10, 256)  # (batch_size, seq_len, hidden_size)
decoder_hidden = torch.randn(32, 256)  # (batch_size, hidden_size)

# 前向传播
context_vector, attention_weights = attention(encoder_outputs, decoder_hidden)

# 打印输出结果
print("Context vector shape:", context_vector.shape)  # (32, 256)
print("Attention weights shape:", attention_weights.shape)  # (32, 10, 1)