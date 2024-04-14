import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.key_layer = nn.Linear(in_features=self.embedding_dim, out_features=self.attention_dim, bias=False)
        self.query_layer = nn.Linear(in_features=self.embedding_dim, out_features=self.attention_dim, bias=False)
        self.value_layer = nn.Linear(in_features=self.embedding_dim, out_features=self.attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        
        key = self.key_layer(embedded)
        query = self.query_layer(embedded)
        value = self.value_layer(embedded)

        weights = (query @ key.transpose(-2, -1)) / self.attention_dim**(0.5)
        tril = torch.tril(torch.ones(embedded.shape[1], embedded.shape[1], dtype=torch.long))

        weights = weights.masked_fill(tril == 0, float('-inf'))
        weights = torch.softmax(weights, dim=-1)

        out = weights @ value

        return torch.round(out, decimals=4)
