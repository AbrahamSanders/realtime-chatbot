import torch
import torch.distributed as dist
from torch import nn
from torch.special import entr
from typing import Optional, Union, List
from transformers.generation import LogitsProcessorList, StoppingCriteriaList, LogitsWarper
from transformers.generation.utils import (
    ContrastiveSearchOutput, 
    ContrastiveSearchDecoderOnlyOutput, 
    ContrastiveSearchEncoderDecoderOutput,
    Seq2SeqLMOutput,
    CausalLMOutputWithPast
)

# >>>>>>>>>>>> START: CUSTOM LOGIC <<<<<<<<<<<<

class TopPProbsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = 0.0, min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_probs, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        
        # scores may be selected from the top-k of the distribution, so fill in the missing probability mass
        cumulative_probs += 1.0-cumulative_probs[:, -1:]

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

# >>>>>>>>>>>> END: CUSTOM LOGIC <<<<<<<<<<<<

def _ranking_fast(
    context_hidden: torch.FloatTensor,
    next_hidden: torch.FloatTensor,
    next_top_k_probs: torch.FloatTensor,
    # >>>>>>>>>>>> START: CUSTOM LOGIC <<<<<<<<<<<<
    alpha: torch.FloatTensor,
    # >>>>>>>>>>>> END: CUSTOM LOGIC <<<<<<<<<<<<
    beam_width: int,
    # >>>>>>>>>>>> START: CUSTOM LOGIC <<<<<<<<<<<<
    penalty_mask: torch.FloatTensor,
    do_sample: bool
    # >>>>>>>>>>>> END: CUSTOM LOGIC <<<<<<<<<<<<
) -> torch.FloatTensor:
    """
    Reranks the top_k candidates based on a degeneration penalty (cosine similarity with previous tokens), as described
    in the paper "A Contrastive Framework for Neural Text Generation". Returns the index of the best candidate for each
    row in the batch.
    """
    # >>>>>>>>>>>> START: CUSTOM LOGIC <<<<<<<<<<<<
    if do_sample:
        zero_probs = next_top_k_probs == 0.0
    # >>>>>>>>>>>> END: CUSTOM LOGIC <<<<<<<<<<<<

    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)  # [B*K, S]
    degeneration_penalty, _ = torch.max(cosine_matrix, dim=-1)  # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]
    
    # >>>>>>>>>>>> START: CUSTOM LOGIC <<<<<<<<<<<<
    alpha = alpha.repeat_interleave(beam_width, dim=0).squeeze(-1)
    contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty * penalty_mask.view(-1)
    # >>>>>>>>>>>> START: END LOGIC <<<<<<<<<<<<
    
    contrastive_score = torch.stack(torch.split(contrastive_score, beam_width))  # [B, K]
    
    # >>>>>>>>>>>> START: CUSTOM LOGIC <<<<<<<<<<<<
    if do_sample:
        contrastive_score[zero_probs] = -1 #-float("Inf")
        #sampling_weights = nn.functional.softmax(contrastive_score, dim=-1)
        sampling_weights = (1+contrastive_score) / (1+contrastive_score).sum(dim=-1, keepdim=True)
        selected_idx = torch.multinomial(sampling_weights, num_samples=1).squeeze(-1)
    # >>>>>>>>>>>> END: CUSTOM LOGIC <<<<<<<<<<<<
    else:
        _, selected_idx = contrastive_score.max(dim=-1)  # [B]
    return selected_idx

def get_contrastive_search_override(self, min_penalty_alpha, max_penalty_alpha, allow_degeneration_token_ids=None, allow_coeff=0.0, 
                                    sample_top_p=0.0, sample_temperature=1.0):

    max_penalty_alpha = max(min_penalty_alpha, max_penalty_alpha)

    @torch.no_grad()
    def contrastive_search(
        input_ids: torch.LongTensor,
        top_k: Optional[int] = 1,
        penalty_alpha: Optional[float] = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        sequential: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[ContrastiveSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **contrastive search** and can
        be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.contrastive_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            top_k (`int`, *optional*, defaults to 1):
                The size of the candidate set that is used to re-rank for contrastive search
            penalty_alpha (`float`, *optional*, defaults to 0):
                The degeneration penalty for contrastive search; activate when it is larger than 0
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            sequential (`bool`, *optional*):
                Switches topk hidden state computation from parallel to sequential to reduce memory if True.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.ContrastiveSearchDecoderOnlyOutput`], [`~generation.ContrastiveSearchEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.ContrastiveSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.ContrastiveSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:
        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        >>> model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> # set pad_token_id to eos_token_id because OPT does not have a PAD token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> input_prompt = "DeepMind Company is"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt")
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
        >>> outputs = model.contrastive_search(
        ...     **input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria
        ... )
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['DeepMind Company is a company that focuses on the development and commercialization of artificial intelligence (AI). DeepMind’s mission is to help people understand and solve problems that are difficult to solve in the world today.\n\nIn this post, we talk about the benefits of deep learning in business and how it']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        sequential = sequential if sequential is not None else self.generation_config.low_memory
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        batch_size = input_ids.shape[0]

        # >>>>>>>>>>>> START: CUSTOM LOGIC <<<<<<<<<<<<
        do_sample = False
        if sample_top_p > 0.0:
            top_p_probs_warper = TopPProbsWarper(sample_top_p)
            do_sample = True
        # >>>>>>>>>>>> END: CUSTOM LOGIC <<<<<<<<<<<<

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # if the first step in the loop, encode all the prefix and obtain: (1) past_key_values;
            # (2) last_hidden_states; (3) logit_for_next_step; (4) update model kwargs for the next step
            if model_kwargs.get("past_key_values") is None:
                # prepare inputs
                model_kwargs["use_cache"] = True
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # encode the given prefix and prepare model inputs; encoder-decoder model process the prefix and save
                # the `encoder_outputs`
                outputs = self(
                    **model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions
                )

                # last decoder hidden states will be used to compute the degeneration penalty (cosine similarity with
                # previous tokens)
                if self.config.is_encoder_decoder:
                    last_hidden_states = outputs.decoder_hidden_states[-1]
                else:
                    last_hidden_states = outputs.hidden_states[-1]

                # next logit for contrastive search to select top-k candidate tokens
                logit_for_next_step = outputs.logits[:, -1, :]

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    standardize_cache_format=True,
                )
                if not sequential:
                    # Expands model inputs top_k times, for batched forward passes (akin to beam search).
                    _, model_kwargs = self._expand_inputs_for_generation(
                        expand_size=top_k, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
                    )

                past_key_values = model_kwargs.get("past_key_values")
                if past_key_values is None:
                    raise ValueError(
                        f"{self.__class__.__name__} does not support caching and therefore **can't** be used "
                        "for contrastive search."
                    )
                elif (
                    not isinstance(past_key_values[0], (tuple, torch.Tensor))
                    or past_key_values[0][0].shape[0] != batch_size
                ):
                    raise ValueError(
                        f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be "
                        "used for contrastive search without further modifications."
                    )

            # contrastive_search main logic start:
            # contrastive search decoding consists of two steps: (1) candidate tokens recall; (2) candidate re-rank by
            # degeneration penalty
            logit_for_next_step = logits_processor(input_ids, logit_for_next_step)
            logit_for_next_step = logits_warper(input_ids, logit_for_next_step)

            # >>>>>>>>>>>> START: CUSTOM LOGIC <<<<<<<<<<<<
            next_probs = nn.functional.softmax(logit_for_next_step / sample_temperature, dim=-1)
            #if do_sample:
            #    top_k_logits, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=top_k)
            #    top_k_top_p_logits = top_p_logits_warper(None, top_k_logits / sample_temperature)
            #    top_k_probs = nn.functional.softmax(top_k_top_p_logits, dim=-1)
            #else:
            top_k_probs, top_k_ids = torch.topk(next_probs, dim=-1, k=top_k)
            if do_sample:
                top_k_probs = top_p_probs_warper(None, top_k_probs)

            if min_penalty_alpha != max_penalty_alpha:
                #next_entropy = entr(next_probs).sum(dim=-1)
                next_expected_prob = (next_probs ** 2).sum(dim=-1, keepdim=True)
                # See: https://www.desmos.com/calculator/bcavepe4bw
                penalty_alpha = min_penalty_alpha + (max_penalty_alpha-min_penalty_alpha) * (1-next_expected_prob)# ** 2
            else:
                penalty_alpha = min_penalty_alpha * torch.ones(
                    batch_size, dtype=next_probs.dtype, device=next_probs.device).unsqueeze(-1)

            #print(f"Penalty-alpha: {penalty_alpha:.2f}")

            penalty_mask = torch.ones_like(top_k_probs)
            if allow_degeneration_token_ids is not None:
                for token_id in allow_degeneration_token_ids:
                    penalty_mask[top_k_ids == token_id] = allow_coeff
            # >>>>>>>>>>>> END: CUSTOM LOGIC <<<<<<<<<<<<

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (logit_for_next_step,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # Replicates the new past_key_values to match the `top_k` candidates
            new_key_values = []
            for layer in model_kwargs["past_key_values"]:
                items = []
                # item is either the key or the value matrix
                for item in layer:
                    if sequential:
                        items.append(item.repeat_interleave(1, dim=0))
                    else:
                        items.append(item.repeat_interleave(top_k, dim=0))
                new_key_values.append(items)
            model_kwargs["past_key_values"] = new_key_values

            if sequential:
                all_outputs = {key: [] for key in outputs}  # defined in first loop iteration
                all_last_hstates, all_hstates, all_logits = [], [], []
                for i in range(top_k):
                    # compute the candidate tokens by the language model and collect their hidden_states
                    next_model_inputs = self.prepare_inputs_for_generation(top_k_ids[:, i].view(-1, 1), **model_kwargs)

                    outputs = self(
                        **next_model_inputs,
                        return_dict=True,
                        output_hidden_states=True,
                        output_attentions=output_attentions,
                    )
                    for key in all_outputs:
                        all_outputs[key].append(outputs[key])

                    if self.config.is_encoder_decoder:
                        next_hidden = outputs.decoder_hidden_states[-1]
                        full_hidden_states = outputs.decoder_hidden_states

                    else:
                        next_hidden = outputs.hidden_states[-1]
                        full_hidden_states = outputs.hidden_states

                    all_last_hstates.append(torch.squeeze(next_hidden, 0))
                    all_hstates.append(full_hidden_states)
                    all_logits.append(outputs.logits[:, -1, :])

                # stack hidden states
                next_hidden = torch.stack([all_last_hstates[i] for i in range(top_k)], dim=0)
                final_full_hstates = [0 for i in range(len(full_hidden_states))]
                for layer in range(len(full_hidden_states)):
                    final_full_hstates[layer] = torch.stack(
                        [torch.squeeze(all_hstates[i][layer], 0) for i in range(top_k)], dim=0
                    )
                full_hidden_states = tuple(final_full_hstates)

                # stack logits
                logits = torch.cat(all_logits, dim=0)

            else:
                # compute the candidate tokens by the language model and collect their hidden_states
                # assembles top_k_ids into batch of size k
                next_model_inputs = self.prepare_inputs_for_generation(top_k_ids.view(-1, 1), **model_kwargs)

                outputs = self(
                    **next_model_inputs,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )
                # name is different for encoder-decoder and decoder-only models
                if self.config.is_encoder_decoder:
                    next_hidden = outputs.decoder_hidden_states[-1]
                    full_hidden_states = outputs.decoder_hidden_states
                else:
                    next_hidden = outputs.hidden_states[-1]
                    full_hidden_states = outputs.hidden_states

                logits = outputs.logits[:, -1, :]

            context_hidden = last_hidden_states.repeat_interleave(top_k, dim=0)

            # compute the degeneration penalty and re-rank the candidates based on the degeneration penalty and the
            # model confidence. Keeping `selected_idx` on CPU enables multi-device contrastive search and doesn't
            # introduce (noticeable) slowdowns on single-device runs.
            selected_idx = _ranking_fast(
                context_hidden, next_hidden, top_k_probs, penalty_alpha, top_k, 
                # >>>>>>>>>>>> START: CUSTOM LOGIC <<<<<<<<<<<<
                penalty_mask, do_sample
                # >>>>>>>>>>>> END: CUSTOM LOGIC <<<<<<<<<<<<
            )
            selected_idx = selected_idx.to("cpu")

            # prepare for the next step: (1) next token_id; (2) past_key_values; (3) last_hidden_states for computing
            # the degeneration penalty; (4) logits for selecting next top-k candidates; (5) selected tokens scores
            # (model confidence minus degeneration penalty); (6) decoder hidden_states
            next_tokens = top_k_ids[range(len(top_k_ids)), selected_idx]
            next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), top_k))
            next_hidden = next_hidden[range(batch_size), selected_idx, :]
            last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)

            next_decoder_hidden_states = ()
            for layer in full_hidden_states:
                layer = torch.stack(torch.split(layer, top_k))[range(batch_size), selected_idx, :]
                next_decoder_hidden_states += (layer,)

            # generate past_key_values cache of only the selected token
            if sequential:
                next_model_input = self.prepare_inputs_for_generation(
                    top_k_ids[:, selected_idx].view(-1, 1), **model_kwargs
                )

                selected_outputs = self(
                    **next_model_input,
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                next_past_key_values = selected_outputs["past_key_values"]

            else:
                next_past_key_values = self._extract_past_from_model_output(outputs, standardize_cache_format=True)
                new_key_values = ()
                for layer in next_past_key_values:
                    items = ()
                    # item is either the key or the value matrix
                    for item in layer:
                        item = torch.stack(torch.split(item, top_k, dim=0))  # [B, K, num_head, seq_len, esz]
                        item = item[range(batch_size), selected_idx, ...]  # [B, num_head, seq_len, esz]
                        items += (item,)
                    new_key_values += (items,)
                next_past_key_values = new_key_values

            logit_for_next_step = torch.stack(torch.split(logits, top_k))[range(batch_size), selected_idx, :]

            # Rebuilds the relevant parts of the model output for the selected token, for use in the next iteration
            if self.config.is_encoder_decoder:
                next_step_cross_attentions = ()
                next_step_decoder_attentions = ()
                if output_attentions:
                    for layer in outputs.cross_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                        next_step_cross_attentions += (layer,)
                    for layer in outputs.decoder_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                        next_step_decoder_attentions += (layer,)
                outputs = Seq2SeqLMOutput(
                    past_key_values=next_past_key_values,
                    decoder_hidden_states=next_decoder_hidden_states,
                    decoder_attentions=next_step_decoder_attentions or None,
                    cross_attentions=next_step_cross_attentions or None,
                )
            else:
                next_step_attentions = ()
                if output_attentions:
                    for layer in outputs.attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                        next_step_attentions += (layer,)
                outputs = CausalLMOutputWithPast(
                    past_key_values=next_past_key_values,
                    hidden_states=next_decoder_hidden_states,
                    attentions=next_step_attentions or None,
                )
            # contrastive_search main logic end

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return ContrastiveSearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return ContrastiveSearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    return contrastive_search
