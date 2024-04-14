import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear_layer_1 = nn.Linear(784, 512)
        self.linear_layer_2 = nn.Linear(512, 10)
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        x = self.linear_layer_1(images)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.2)(x)
        x = self.linear_layer_2(x)
        x = nn.Sigmoid()(x)

        return x
