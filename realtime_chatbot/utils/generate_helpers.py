from transformers.generation import StoppingCriteria
import torch

class CompletionAndResponseStoppingCriteria(StoppingCriteria):
    def __init__(self, turn_switch_token_id: int, max_response_length: int):
        self.turn_switch_token_id = turn_switch_token_id
        self.max_response_length = max_response_length
        self.completion_length = 0
        self.response_length = 0
        self.completion_done = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:        
        if input_ids[0, -1] == self.turn_switch_token_id:
            if self.completion_done:
                return True
            self.completion_done = True
            self.response_length = 1
        elif not self.completion_done:
            self.completion_length += 1
        else:
            self.response_length += 1
            if self.response_length >= self.max_response_length:
                return True
        return False