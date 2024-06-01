import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear transformations for query, key, and value
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)
        
        # Reshape query, key, and value for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to value
        attended_values = torch.matmul(attention_weights, value)
        
        # Concatenate and reshape attended values
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Linear transformation for output
        output = self.output_linear(attended_values)
        
        return output
    
    
# Create input tensors
batch_size = 2
seq_length = 5
d_model = 64
num_heads = 4

query = torch.randn(batch_size, seq_length, d_model)
key = torch.randn(batch_size, seq_length, d_model)
value = torch.randn(batch_size, seq_length, d_model)

# Create multi-head attention module
multihead_attention = MultiHeadAttention(d_model, num_heads)

# Compute multi-head attention
output = multihead_attention(query, key, value)

print(output.shape)  # Output shape: (batch_size, seq_length, d_model)