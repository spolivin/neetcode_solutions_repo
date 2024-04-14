import torch
import torch.nn as nn
from torchtyping import TensorType


class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.model_dim = model_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.model_dim)
        self.position_embedding_table = nn.Embedding(self.context_length, self.model_dim)
        self.transformer_blocks = nn.Sequential(*[self.TransformerBlock(self.model_dim, self.num_heads) for _ in range(self.num_blocks)])
        self.layernorm_final = nn.LayerNorm(self.model_dim)
        self.langmod_head = nn.Linear(self.model_dim, self.vocab_size)

    def forward(self, context: TensorType[int]) -> TensorType[float]:
        torch.manual_seed(0)
        
        token_embeddings = self.token_embedding_table(context)
        position_embeddings = self.position_embedding_table(torch.arange(context.shape[1]))
        
        x = token_embeddings + position_embeddings
        x = self.transformer_blocks(x)
        x = self.layernorm_final(x)
        x = nn.functional.softmax(self.langmod_head(x), dim=-1)

        return torch.round(x, decimals=4)

    
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    torch.manual_seed(0)
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)
                
                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    k = self.key_gen(embedded)
                    q = self.query_gen(embedded)
                    v = self.value_gen(embedded)

                    scores = q @ torch.transpose(k, 1, 2)
                    context_length, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim ** 0.5)

                    lower_triangular = torch.tril(torch.ones(context_length, context_length))
                    mask = lower_triangular == 0
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim = 2)

                    return scores @ v
                
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)
                self.att_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                head_outputs = []
                for head in self.att_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim = 2)
                return concatenated
        
        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            embedded = embedded + self.attention(self.first_norm(embedded))
            embedded = embedded + self.linear_network(self.second_norm(embedded))
            return embedded
