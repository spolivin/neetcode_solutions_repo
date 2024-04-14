import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        for i in range(new_chars):

            
            context_cond = context[:, -context_length:]
            logits = model(context_cond)
            logits = logits[:, -1, :]
            probas = nn.functional.softmax(logits, dim=-1)
            
            generator.set_state(initial_state)
            idx_next = torch.multinomial(probas, num_samples=1, generator=generator)
            idx = torch.cat((context, idx_next), dim=1)
        
        decode = lambda l: ''.join([int_to_char[i] for i in l])

        return decode(idx[0].tolist()[1:])
