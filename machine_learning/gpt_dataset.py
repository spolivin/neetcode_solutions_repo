import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        torch.manual_seed(0)
        
        vocab = set()
        for word in raw_dataset.split():
            vocab.add(word)
        char2id = {s: i for i, s in enumerate(vocab)}
        id2char = {i: s for i, s in enumerate(vocab)}
        encode = lambda s: [char2id[c] for c in s.split()]
        decode = lambda l: [id2char[i] for i in l]

        data = torch.tensor(encode(raw_dataset), dtype=torch.long)
        ix = torch.randint(len(data) - context_length, (batch_size,))
        X = torch.stack([data[i:i+context_length] for i in ix]).tolist()
        Y = torch.stack([data[i+1:i+context_length+1] for i in ix]).tolist()

        X = [decode(seq) for seq in X]
        Y = [decode(seq) for seq in Y]
        
        return X, Y
