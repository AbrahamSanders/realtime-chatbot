import torch
from transformers import Trainer, DataCollatorWithPadding
from torch.special import entr

class AnchoredDistillingTrainer(Trainer):
    def __init__(self, anchor_model, anchor_logits_dimension, *args, 
                 anchor_loss_weight=5.0, kl_div_temperature=2.0, lm_loss_weight=1.0,
                 embed_cosine_loss_weight=1.0, freeze_pos_embeds=True, 
                 use_token_anchor_loss_weighting=False, use_token_lm_loss_weighting=False, 
                 lm_loss_sigmoid_coeff=2.0, **kwargs):
        
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
        self.kl_div_temperature = kl_div_temperature
        self.lm_loss_weight = lm_loss_weight
        self.embed_cosine_loss_weight = embed_cosine_loss_weight
        if freeze_pos_embeds:
            kwargs['model'].model.decoder.embed_positions.weight.requires_grad = False
        
        self.use_token_anchor_loss_weighting = use_token_anchor_loss_weighting
        self.use_token_lm_loss_weighting = use_token_lm_loss_weighting
        self.lm_loss_sigmoid_coeff = lm_loss_sigmoid_coeff

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.kl_div_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(reduction="none")

    def get_token_weights(self, logits, labels):
        if not self.use_token_anchor_loss_weighting or not self.use_token_lm_loss_weighting:
            identity_weights = torch.ones_like(labels, dtype=logits.dtype)
        if not self.use_token_anchor_loss_weighting and not self.use_token_lm_loss_weighting:
            return identity_weights, identity_weights.clone()

        # entropy
        token_probs = torch.nn.functional.softmax(logits, dim=-1)
        token_entropies = entr(token_probs).sum(dim=-1)
        token_expected_probs = torch.exp(-token_entropies)

        # crossentropy
        token_crossentropies = self.cross_entropy_loss(logits, labels)
        token_correct_probs = torch.exp(-token_crossentropies)

        # weighting
        if self.use_token_anchor_loss_weighting:
            token_anchor_weights = torch.maximum(1-token_expected_probs, token_correct_probs)
        else:
            token_anchor_weights = identity_weights

        if self.use_token_lm_loss_weighting:
            token_lm_weights = torch.sigmoid(self.lm_loss_sigmoid_coeff*torch.pi*(token_expected_probs-token_correct_probs))
        else:
            token_lm_weights = identity_weights

        return token_anchor_weights, token_lm_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # -- (1) Setup for All Training Objectives ----------------------------------------------------------------
        # construct a position mask for all training objectives to avoid applying losses at positions that predict padding
        position_mask = inputs.attention_mask.to(torch.bool)
        # the last non-padding input position is the first to predict padding, so exclude it too
        last_non_padding_selector = (torch.arange(position_mask.shape[0]).unsqueeze(1).to(position_mask.device), 
                                     inputs.attention_mask.sum(dim=-1, keepdim=True)-1)
        position_mask[last_non_padding_selector] = False
        # when selecting shifted labels, the sequence dimension will be one less than in input_ids.
        labels_mask = position_mask[:, :-1]

        # get LM logits, shifted labels, and hidden states at last layer
        outputs = model(input_ids = inputs.input_ids, 
                        attention_mask = inputs.attention_mask, 
                        output_hidden_states=True, 
                        return_dict=True)
        lm_logits = outputs.logits[position_mask].contiguous()
        lm_labels = inputs.input_ids[..., 1:][labels_mask].contiguous()
        lm_hidden_states = outputs.hidden_states[-1][position_mask]

        # get anchor logits & hidden states at last layer
        with torch.no_grad():
            anchor_outputs = self.anchor_model(input_ids = inputs.input_ids.to(self.anchor_model.device), 
                                               attention_mask = inputs.attention_mask.to(self.anchor_model.device),
                                               output_hidden_states=True,
                                               return_dict=True)
            anchor_logits = anchor_outputs.logits.to(lm_logits.device)
            anchor_logits = anchor_logits[position_mask].contiguous()
            anchor_hidden_states = anchor_outputs.hidden_states[-1].to(lm_logits.device)
            anchor_hidden_states = anchor_hidden_states[position_mask]

        # get anchor weights for each token (e.g., how much should we apply the KL & cosine losses for each token?)
        # and also get lm weights for each token (e.g. how much should we apply crossentropy supervision for each token?)
        token_anchor_weights, token_lm_weights = self.get_token_weights(anchor_logits, lm_labels)
        # always set anchor weight to 0 for input positions with additional special tokens, since the anchor model
        # has not seen them during pre-training and thus has no useful representation for them.
        special_input_positions = (inputs.input_ids >= self.anchor_logits_dimension)[position_mask]
        token_anchor_weights[special_input_positions] = 0.
        token_lm_weights[special_input_positions] = 1.

        # -- (2) Compute LM Loss ----------------------------------------------------------------------------------
        lm_loss = self.cross_entropy_loss(lm_logits, lm_labels)
        lm_loss *= token_lm_weights
        # weighted 'mean' reduction
        lm_loss = lm_loss.sum() / token_lm_weights.sum()

        outputs.loss = self.lm_loss_weight * lm_loss

        # -- (3) Compute KL Divergence Loss -----------------------------------------------------------------------
        # for KL objective, we only care about vocab entries that existed in the anchor model during pre-training.
        lm_logits_for_kl = lm_logits[..., :self.anchor_logits_dimension]
        anchor_logits_for_kl = anchor_logits[..., :self.anchor_logits_dimension]

        # original: https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/distiller.py#L410
        lm_log_probs = torch.nn.functional.log_softmax(lm_logits_for_kl / self.kl_div_temperature, dim=-1)
        anchor_log_probs = torch.nn.functional.log_softmax(anchor_logits_for_kl / self.kl_div_temperature, dim=-1)
        kl_div_loss = self.kl_div_loss(lm_log_probs, anchor_log_probs).sum(dim=-1)
        kl_div_loss *= token_anchor_weights
        # weighted 'batchmean' reduction (not 'mean' reduction since we already summed over the vocab dimension)
        kl_div_loss = kl_div_loss.sum() / token_anchor_weights.sum()
        kl_div_loss *= (self.kl_div_temperature) ** 2
        
        outputs.loss += self.anchor_loss_weight * kl_div_loss

        # -- (4) Compute Cosine Similarity Loss -------------------------------------------------------------------
        if self.embed_cosine_loss_weight > 0.0:
            embed_cosine_targets = lm_hidden_states.new(lm_hidden_states.shape[0]).fill_(1)
            embed_cosine_loss = self.cosine_loss(lm_hidden_states, anchor_hidden_states, embed_cosine_targets)
            embed_cosine_loss *= token_anchor_weights
            # weighted 'mean' reduction
            embed_cosine_loss = embed_cosine_loss.sum() / token_anchor_weights.sum()
                
            outputs.loss += self.embed_cosine_loss_weight * embed_cosine_loss

        return (outputs.loss, outputs) if return_outputs else outputs.loss

class DataCollatorWithPaddingAndLabels(DataCollatorWithPadding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["input_ids"]
        return batch