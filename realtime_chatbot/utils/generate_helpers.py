from transformers.generation import StoppingCriteria
import torch

class CompletionAndResponseStoppingCriteria(StoppingCriteria):
    def __init__(self, turn_switch_token_id: int):
        self.turn_switch_token_id = turn_switch_token_id
        self.completion_length = 0
        self.completion_done = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:        
        if input_ids[0, -1] == self.turn_switch_token_id:
            if self.completion_done:
                return True
            self.completion_done = True
        elif not self.completion_done:
            self.completion_length += 1
        return False