import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.vocabulary_size = vocabulary_size
        self.embedding_layer = nn.Embedding(self.vocabulary_size, 16)
        self.linear_layer = nn.Linear(16, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        
        embedded = self.embedding_layer(x)
        embedded_averaged = torch.mean(embedded, dim=1)
        
        return self.activation(self.linear_layer(embedded_averaged))
