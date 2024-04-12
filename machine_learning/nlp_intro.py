from typing import List

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        
        vocabulary = set()
        
        heaps = [positive, negative]
        for heap in heaps:
            for sentence in heap:
                for word in sentence.split():
                    vocabulary.add(word)
        
        vocabulary = sorted(vocabulary)
        char2id = {s: i+1 for i, s in enumerate(vocabulary)}
        
        outputs = []
        
        for heap in heaps:
            for sentence in heap:
                enc = []
                for word in sentence.split():
                    enc.append(char2id[word])
                outputs.append(torch.tensor(enc))
        
        return nn.utils.rnn.pad_sequence(outputs, batch_first=True)
