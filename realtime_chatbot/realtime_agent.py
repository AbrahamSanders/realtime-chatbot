from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

from .identity import Identity

class RealtimeAgent_Resources:
    def __init__(self, modelpath="rtchat-2.7b/checkpoint-700"):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)
        self.tokenizer.truncation_side = "left"
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(modelpath)
        self.model = self.model.to(self.device)

    def create_agent(self, identities=None):
        return RealtimeAgent(self, identities)

class RealtimeAgent:
    def __init__(self, resources=None, identities=None):
        if resources is None:
            resources = RealtimeAgent_Resources()
        self.resources = resources

        if identities is None:
            identities = Identity.default_identities()
        self.identities = identities

        self.tokenizer_max_length = None
        if (not self.resources.tokenizer.model_max_length or self.resources.tokenizer.model_max_length > 9999) \
              and hasattr(self.resources.model.config, "max_position_embeddings"):
            self.tokenizer_max_length = self.resources.model.config.max_position_embeddings

        self.generate_kwargs = {
            "pad_token_id": self.resources.tokenizer.pad_token_id,
            "eos_token_id": self.resources.tokenizer.eos_token_id,
            "max_new_tokens": 50,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50,
            "temperature": 1.6,
            "num_beams": 2,
            "early_stopping": True
        }

        self.reset()

    def _generate(self, sequence, stopping_criteria=None, **generate_kwargs_overrides):
        # Configure generation params
        generate_kwargs = self.generate_kwargs.copy()
        if generate_kwargs_overrides:
            generate_kwargs.update(generate_kwargs_overrides)

        # Tokenize sequence
        inputs = self.resources.tokenizer(
            sequence, truncation=True, max_length=self.tokenizer_max_length, return_tensors="pt"
        ).to(self.resources.device)

        # Generate
        generate_result = self.resources.model.generate(
            input_ids=inputs.input_ids,
            attention_mask = inputs.attention_mask,
            return_dict_in_generate=True,
            **generate_kwargs
        )
        result_ids = generate_result.sequences.cpu()

        # Decode and return results
        results = []
        stopping_matches = []
        for i in range(result_ids.shape[0]):
            response_start_idx = inputs.input_ids.shape[-1]
            generated_text = self.resources.tokenizer.decode(result_ids[i, response_start_idx:], 
                                                             skip_special_tokens=False)
            generated_text = generated_text.replace(self.resources.tokenizer.pad_token, "")
            generated_text = generated_text.replace(self.resources.tokenizer.eos_token, "")
            # If a regex stopping criteria is specified, check for it and discard anything
            # after its match ends
            if stopping_criteria is not None:
                match = re.search(stopping_criteria, generated_text, re.IGNORECASE)
                stopping_matches.append(match)
                if match:
                    generated_text = generated_text[:match.end()]

            results.append(generated_text)

        gen_results = results if len(results) > 1 else results[0]
        return gen_results

    def reset(self):
        self.sequence = ""
        for identity, info in self.identities.items():
            self.sequence += f"<participant> {identity} (name: {info.name}, age: {info.age}, sex: {info.sex}) "
        self.sequence += "<dialog> "

    def execute(self, identity="S2"):
        self.sequence += f"{identity}:"
        self.sequence += self._generate(self.sequence, stopping_criteria=".+?(?=S1:)")
        if not self.sequence.endswith(" "):
            self.sequence += " "

    def append_input(self, input, identity="S1"):
        self.sequence += f"{identity}: {input} "