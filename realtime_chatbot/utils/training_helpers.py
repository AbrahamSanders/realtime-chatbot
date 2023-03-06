import torch
from transformers import Trainer, DataCollatorWithPadding
from torch.special import entr

class AnchoredDistillingTrainer(Trainer):
    def __init__(self, anchor_model, anchor_logits_dimension, *args, 
                 anchor_loss_weight=5.0, lm_loss_weight=1.0, kl_div_temperature=2.0,
                 embed_cosine_loss_weight=1.0, freeze_pos_embeds=True, 
                 token_anchor_weighting="none", token_entr_temperature=0.3, **kwargs):
        
        self.anchor_model = anchor_model
        self.anchor_model.eval()
        super().__init__(*args, **kwargs)

        #if self.anchor_model.device != kwargs['model'].device:
        #     self.anchor_model = self.anchor_model.to(kwargs['model'].device)
        # TODO: probably won't work on multi-gpu training. This allows the anchor model to be on a different 
        #       GPU than where the fine-tuning is happening.
        self.args._n_gpu = 1
        anchor_device_idx = 1 if torch.cuda.device_count() > 1 else 0
        self.anchor_model = self.anchor_model.to(torch.device(f"cuda:{anchor_device_idx}" if torch.cuda.is_available() else "cpu"))
        if "args" in kwargs and kwargs["args"].fp16:
            self.anchor_model = self.anchor_model.half()

        self.anchor_logits_dimension = anchor_logits_dimension
        self.anchor_loss_weight = anchor_loss_weight
        self.lm_loss_weight = lm_loss_weight
        self.kl_div_temperature = kl_div_temperature
        self.embed_cosine_loss_weight = embed_cosine_loss_weight
        if freeze_pos_embeds:
            kwargs['model'].model.decoder.embed_positions.weight.requires_grad = False
        self.token_anchor_weighting_fn = self.get_token_anchor_weighting_fn(
            token_anchor_weighting, token_entr_temperature
        )

        self.kl_div_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(reduction="none")

    def get_token_anchor_weighting_fn(self, token_anchor_weighting, token_entr_temperature):
        valid_weighting_options = ["none", "inv_entr", "inv_exp_entr", "sigmoid_neg_entr"]
        if token_anchor_weighting not in valid_weighting_options:
            raise ValueError(f"token_anchor_weighting must be one of {valid_weighting_options}.")
        if token_anchor_weighting == "none":
            return None
        elif token_anchor_weighting == "inv_entr":
            weighting_fn = lambda token_entropies: 1 / (1 + token_entropies)
        elif token_anchor_weighting == "inv_exp_entr":
            weighting_fn = lambda token_entropies: torch.exp(-token_entropies)
        else:
            weighting_fn = lambda token_entropies: 2 / (1 + torch.exp(token_entropies))

        def token_anchor_weighting_fn(anchor_logits):
            token_probs = torch.nn.functional.softmax(anchor_logits / token_entr_temperature, dim=-1)
            token_entropies = entr(token_probs).sum(dim=-1)
            token_anchor_weights = weighting_fn(token_entropies)
            return token_anchor_weights
        
        return token_anchor_weighting_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        # construct a logits mask for anchoring objectives to avoid anchoring positions with padding or additional special tokens
        # (input is chunked and not padded during pre-training, and additional special tokens didn't exist yet).
        logits_mask = inputs.attention_mask.to(torch.bool)
        logits_mask[inputs.input_ids >= self.anchor_logits_dimension] = False

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        lm_loss = outputs.loss
        lm_logits = outputs.logits[logits_mask][..., :self.anchor_logits_dimension]
        lm_hidden_states = outputs.hidden_states[-1][logits_mask]
        with torch.no_grad():
            anchor_outputs = self.anchor_model(input_ids = inputs.input_ids.to(self.anchor_model.device), 
                                               attention_mask = inputs.attention_mask.to(self.anchor_model.device),
                                               output_hidden_states=True,
                                               return_dict=True)
            anchor_logits = anchor_outputs.logits.to(lm_logits.device)
            anchor_logits = anchor_logits[logits_mask][..., :self.anchor_logits_dimension]
            anchor_hidden_states = anchor_outputs.hidden_states[-1].to(lm_logits.device)
            anchor_hidden_states = anchor_hidden_states[logits_mask]

        token_anchor_weights = None
        if self.token_anchor_weighting_fn is not None:
            token_anchor_weights = self.token_anchor_weighting_fn(anchor_logits)

        # original: https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/distiller.py#L410
        lm_log_probs = torch.nn.functional.log_softmax(lm_logits / self.kl_div_temperature, dim=-1)
        anchor_log_probs = torch.nn.functional.log_softmax(anchor_logits / self.kl_div_temperature, dim=-1)
        kl_div_loss = self.kl_div_loss(lm_log_probs, anchor_log_probs).sum(dim=-1)
        if token_anchor_weights is not None:
            kl_div_loss *= token_anchor_weights
        kl_div_loss = kl_div_loss.mean() # 'batchmean' reduction (since we already summed over the vocab dimension)
        kl_div_loss *= (self.kl_div_temperature) ** 2
        
        outputs.loss = self.anchor_loss_weight * kl_div_loss
        outputs.loss += self.lm_loss_weight * lm_loss

        if self.embed_cosine_loss_weight > 0.0:
            embed_cosine_targets = lm_hidden_states.new(lm_hidden_states.shape[0]).fill_(1)
            embed_cosine_loss = self.cosine_loss(lm_hidden_states, anchor_hidden_states, embed_cosine_targets)
            if token_anchor_weights is not None:
                embed_cosine_loss *= token_anchor_weights
                # weighted 'mean' reduction
                embed_cosine_loss = embed_cosine_loss.sum() / token_anchor_weights.sum()
            else:
                embed_cosine_loss = embed_cosine_loss.mean() # standard 'mean' reduction
                
            outputs.loss += self.embed_cosine_loss_weight * embed_cosine_loss

        return (outputs.loss, outputs) if return_outputs else outputs.loss

class DataCollatorWithPaddingAndLabels(DataCollatorWithPadding):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
    
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["input_ids"]
        return batch