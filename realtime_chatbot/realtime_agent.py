from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
from transformers.generation import StoppingCriteriaList
import torch
import re
import math
from time import sleep
from datetime import datetime
from collections import deque

from .identity import Identity
from .utils import queue_helpers
from .utils.generate_helpers import CompletionAndResponseStoppingCriteria
from .dynamic_contrastive import get_contrastive_search_override

class RealtimeAgent_Resources:
    def __init__(self, modelpath="AbrahamSanders/opt-2.7b-realtime-chat-v2", device=None, 
                 use_fp16=True, override_contrastive_search=True):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)
        self.tokenizer.truncation_side = "left"
        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # Model
        if use_fp16:
            self.model = AutoModelForCausalLM.from_pretrained(modelpath, torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(modelpath)

        self.model = self.model.to(self.device)

        if override_contrastive_search:
            self.model.contrastive_search = get_contrastive_search_override(self.model, 0.005, 1.0, sample_top_p=0.8)

    def create_agent(self, config=None):
        return RealtimeAgent(resources=self, config=config)

class RealtimeAgentConfig:
    def __init__(self, identities=None, user_identity="S1", agent_identity="S2", 
                 interval=0.8, max_history_words=100, max_agent_pause_duration=10.0, 
                 random_state=None, prevent_special_token_generation=False, summary=None,
                 add_special_pause_token=False, predictive_lookahead=True, similarity_threshold=0.8):
        if identities is None:
            identities = Identity.default_identities()
        self.identities = identities
        self.user_identity = user_identity
        self.agent_identity = agent_identity
        self.interval = interval
        self.max_history_words = max_history_words
        self.max_agent_pause_duration = max_agent_pause_duration
        self.random_state = random_state
        self.prevent_special_token_generation = prevent_special_token_generation
        self.summary = summary
        self.add_special_pause_token = add_special_pause_token
        self.predictive_lookahead = predictive_lookahead
        self.similarity_threshold = similarity_threshold

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class PredictedCompletion:
    def __init__(self):
        self.context_pos = None
        self.completion = None
        self.embedding = None
        self.response = None
        self.response_speaker = None

class RealtimeAgent:
    def __init__(self, resources=None, config=None):
        if resources is None:
            resources = RealtimeAgent_Resources()
        self.resources = resources

        if config is None:
            config = RealtimeAgentConfig()
        self.config = config

        self.tokenizer_max_length = None
        if (not self.resources.tokenizer.model_max_length or self.resources.tokenizer.model_max_length > 9999) \
              and hasattr(self.resources.model.config, "max_position_embeddings"):
            self.tokenizer_max_length = self.resources.model.config.max_position_embeddings

        self.generate_kwargs = {
            "pad_token_id": self.resources.tokenizer.pad_token_id,
            "eos_token_id": self.resources.tokenizer.eos_token_id,
            "max_new_tokens": 7,
            "penalty_alpha": 0.1, # penalty_alpha is overridden by dynamic contrastive search, but needs to be set to something > 0
            "top_k": 8
        }
        if self.config.prevent_special_token_generation:
            bad_words_ids = self.resources.tokenizer(["\n", self.resources.tokenizer.eos_token,
                "participants:", " participants:", "Participants:", " Participants:",
                "summary:", " summary:", "Summary:", " Summary:",
                "transcript:", " transcript:", "Transcript:", " Transcript:"], 
                add_prefix_space=False, add_special_tokens=False).input_ids
            self.generate_kwargs["bad_words_ids"] = bad_words_ids

        self.any_identity_regex = re.compile(r"S\d+?")
        self.any_identity_with_incomplete_regex = re.compile(rf" (?:{self.any_identity_regex.pattern}|S\Z)")
        self.agent_turn_switch_regex = re.compile(rf"(?<={self.config.agent_identity}:).+?(?= {self.any_identity_regex.pattern}|\Z)")
        self.speaker_continue_regex = re.compile(rf".+?(?= {self.any_identity_regex.pattern}|\Z)")
        self.pause_regex = re.compile(r"(?:<p> )?\((\d*?\.\d*?)\)")
        self.pause_at_end_regex = re.compile(rf"{self.pause_regex.pattern}\Z")
        self.incomplete_pause_regex = re.compile(r"(?:<p> ?\(?|\()\d*?\.?\d*?\Z")
        self.end_pause_regex = re.compile(r"\)")
        self.input_segments_regex = re.compile(" (?=[~*])")
        self.sequence_split_regex = re.compile(rf"\s(?={self.any_identity_regex.pattern})")

        self.prefix_length = None
        
        self.reset()

    def _generate(self, sequence, stopping_criteria=None, take_tokens=None, return_generate_output=False, 
                  always_return_list=False, **generate_kwargs_overrides):
        # Configure generation params
        generate_kwargs = self.generate_kwargs.copy()
        if generate_kwargs_overrides:
            generate_kwargs.update(generate_kwargs_overrides)
        if isinstance(stopping_criteria, StoppingCriteriaList):
            generate_kwargs["stopping_criteria"] = stopping_criteria
            stopping_criteria = None

        input_is_list = isinstance(sequence, (list, tuple))
        if input_is_list:
            always_return_list = True
            
        if input_is_list or isinstance(sequence, str):
            # Tokenize sequence with left padding
            original_padding_side = self.resources.tokenizer.padding_side
            self.resources.tokenizer.padding_side = "left"
            inputs = self.resources.tokenizer(
                sequence, padding=True, truncation=True, max_length=self.tokenizer_max_length, return_tensors="pt"
            ).to(self.resources.device)
            self.resources.tokenizer.padding_side = original_padding_side
        else:
            inputs = {"input_ids": sequence.to(self.resources.device)}

        generate_kwargs.update(inputs)

        # Generate
        if self.config.random_state is not None:
            set_seed(self.config.random_state)

        outputs = self.resources.model.generate(return_dict_in_generate=True, **generate_kwargs)

        # Decode and return results
        results = []
        for i in range(outputs.sequences.shape[0]):
            response_start_idx = inputs["input_ids"].shape[-1]
            response_end_idx = response_start_idx + take_tokens if take_tokens is not None else outputs.sequences.shape[-1]
            generated_text = self.resources.tokenizer.decode(outputs.sequences[i, response_start_idx:response_end_idx], 
                                                             skip_special_tokens=False)
            generated_text = generated_text.replace(self.resources.tokenizer.pad_token, "")
            generated_text = generated_text.replace(self.resources.tokenizer.eos_token, "")
            # If a regex stopping criteria is specified, check for it and discard anything
            # after its match ends
            if stopping_criteria is not None:
                match = re.search(stopping_criteria, generated_text)
                if match:
                    generated_text = generated_text[:match.end()]

            results.append(generated_text)

        gen_results = results if always_return_list or len(results) > 1 else results[0]
        if return_generate_output:
            gen_results = (gen_results, outputs)
        return gen_results

    def _set_current_speaker(self, identity):
        self.sequence += f" {identity}:"
        self.current_speaker = identity
        self.last_prediction = None

    def _trim_sequence(self):
        dialog_history = self.sequence[self.prefix_length:]
        history_split = dialog_history.split()
        if len(history_split) > self.config.max_history_words:
            history_split = history_split[-self.config.max_history_words:]
            for i in range(len(history_split)):
                if re.match(self.any_identity_regex, history_split[i]):
                    break
            i = 0 if i == len(history_split)-1 else i
            trimmed_history = " ".join(history_split[i:])
            prev_sequence_length = len(self.sequence)
            self.sequence = f"{self.sequence[:self.prefix_length]}{trimmed_history}"
            if self.partial_pos > -1:
                self.partial_pos += len(self.sequence)-prev_sequence_length
            if self.last_prediction is not None:
                self.last_prediction.context_pos += len(self.sequence)-prev_sequence_length

    def _incrementing_pause(self, seconds_since_last_cycle):
        # get previous pause duration (if any)
        pause_match = re.search(self.pause_at_end_regex, self.sequence[-10:])
        if pause_match:
            pause_duration = self.config.interval if pause_match[1] == "." else float(pause_match[1])
            pause_match_len = pause_match.end()-pause_match.start()+1
            self.sequence = self.sequence[:-pause_match_len]
        else:
            pause_duration = 0.0

        # increment and return
        pause_duration += seconds_since_last_cycle
        pause_prefix = "<p> " if self.config.add_special_pause_token else ""
        return f" {pause_prefix}({pause_duration:.1f})"
        
    def _set_prefix(self):
        prefix = "Participants: "
        for identity, info in self.config.identities.items():
            prefix += f"{identity} (name: {info.name}, age: {info.age}, sex: {info.sex}), "
        prefix = prefix.rstrip(", ") + "; "
        if self.config.summary:
            prefix += f"Summary: {self.config.summary}; "
        prefix += "Transcript:"
        prev_prefix_length = None
        if len(self.sequence) > 0 and self.prefix_length is not None:
            self.sequence = f"{prefix} {self.sequence[self.prefix_length:]}"
            prev_prefix_length = self.prefix_length
        else:
            self.sequence = prefix
        self.prefix_length = len(prefix)+1
        if prev_prefix_length is not None:
            if self.partial_pos > -1:
                self.partial_pos += self.prefix_length-prev_prefix_length
            if self.last_prediction is not None:
                self.last_prediction.context_pos += self.prefix_length-prev_prefix_length

    def _get_next_slice_index(self, str, i):
        if i < len(str):
            for pos in range(i, len(str)):
                if str[pos] == " ":
                    return pos
        return len(str)

    def _update_sequence_from_input(self, next_input):
        sequence_changed = False
        if not next_input:
            return sequence_changed
        # First, clear out the previous partial utterance segment (if exists)
        utterances_after_partial_pos = []
        if self.partial_pos > -1:
            # Locate all turns taken after the partial utterance has started
            utterances_after_partial_pos = re.split(self.sequence_split_regex, self.sequence[self.partial_pos:])
            utterances_after_partial_pos = list(filter(None, utterances_after_partial_pos))
            # Reset the sequence to the position where the partial utterance begins
            self.sequence = self.sequence[:self.partial_pos]
            sequence_changed = True
            self.partial_pos = -1
        # Next, add the new segments to the sequence, discarding intermediate partial segments.
        new_segments = re.split(self.input_segments_regex, next_input)
        for i, seg in enumerate(new_segments):
            if seg and (seg.startswith("*") or i == len(new_segments)-1):
                if seg.startswith("~"):
                    self.partial_pos = len(self.sequence)
                seg_text = seg[1:]
                # Iterate through all turns taken after the previous partial utterance started.
                # Replace user utterances with words from the new segment while carrying non-user 
                # utterances over intact.
                deferred_start = False
                for j, utt in enumerate(utterances_after_partial_pos):
                    identity_match = re.match(self.any_identity_regex, utt)
                    if identity_match and not utt.startswith(self.config.user_identity):
                        # if this utterance is the last turn after the previous partial utterance started
                        # and there are two or less words left in the new segment, append those words to the previous
                        # utterance instead of letting them hang off the end of the sequence as a new user utterance.
                        # Note: assumes the previous utterance belongs to user. Will need to revisit for multiparty.
                        if (j == len(utterances_after_partial_pos)-1 and seg_text and seg_text.count(" ") < 2
                                                                     and (j > 1 or not deferred_start)):
                            self.sequence += f" {seg_text}"
                            seg_text = ""
                        # carry non-user utterance over intact
                        self._set_current_speaker(identity_match[0])
                        utt = utt[identity_match.end()+1:].lstrip()
                        self.sequence += f" {utt}"
                        sequence_changed = True
                    elif seg_text:
                        # replace user utterance with words (of same approximate length)
                        # from the new segment
                        has_user_identity = utt.startswith(self.config.user_identity)
                        if has_user_identity:
                            utt = utt[len(self.config.user_identity)+1:]
                        utt = utt.lstrip()
                        next_slice_idx = self._get_next_slice_index(seg_text, len(utt))
                        seg_text_slice = seg_text[:next_slice_idx]
                        # if this utterance is the first turn after the previous partial utterance started
                        # and there are two or less words to place before the next turn, defer these words to appear after it.
                        if j == 0 and len(utterances_after_partial_pos) > 1 and seg_text_slice.count(" ") < 2:
                            deferred_start = True
                            continue
                        if has_user_identity:
                            self._set_current_speaker(self.config.user_identity)
                        self.sequence += f" {seg_text_slice}"
                        sequence_changed = True
                        seg_text = seg_text[next_slice_idx:].lstrip()
                utterances_after_partial_pos.clear()
                # any remaining text in the new segment is appended to the end of the sequence
                if seg_text:
                    if self.current_speaker != self.config.user_identity:
                        self._set_current_speaker(self.config.user_identity)
                    self.sequence += f" {seg_text}"
                    sequence_changed = True
        return sequence_changed

    def _predict_completion(self, last_prediction):
        last_pred_embedding_exists = last_prediction is not None and last_prediction.embedding is not None
        if last_pred_embedding_exists and last_prediction.context_pos < len(self.sequence):
            last_pred_context, actual_completion = self.sequence[:last_prediction.context_pos], self.sequence[last_prediction.context_pos:]
            last_pred_context_tokens = self.resources.tokenizer.encode(last_pred_context, return_tensors="pt").to(self.resources.device)
            actual_completion_tokens = self.resources.tokenizer.encode(actual_completion, return_tensors="pt", add_special_tokens=False).to(self.resources.device)
            input_ids = torch.cat((last_pred_context_tokens, actual_completion_tokens), dim=1)
            with torch.no_grad():
                outputs = self.resources.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
                actual_completion_embedding = outputs.hidden_states[-1][:, last_pred_context_tokens.shape[-1]:].mean(dim=1)
                similarity_with_last_pred = torch.cosine_similarity(last_prediction.embedding, actual_completion_embedding, dim=-1).item()
        else:
            input_ids = self.resources.tokenizer.encode(self.sequence, return_tensors="pt").to(self.resources.device)
            similarity_with_last_pred = 1.0 if last_pred_embedding_exists else -1.0

        turn_switch_token = " S"
        turn_switch_token_id = self.resources.tokenizer(turn_switch_token, add_special_tokens=False).input_ids[0]
        if similarity_with_last_pred >= self.config.similarity_threshold:
            # If last prediction is similar enough to the actual completion, no need to make a new prediction. Just check if we're at a turn-switch.
            prediction = last_prediction
            next_token = self._generate(input_ids, max_new_tokens=1)
            is_turn_switch = next_token == turn_switch_token
        else:
            # Otherwise, make a new utterance completion prediction.
            stopping_criteria = StoppingCriteriaList([CompletionAndResponseStoppingCriteria(turn_switch_token_id)])
            _, outputs = self._generate(
                input_ids, stopping_criteria=stopping_criteria, 
                return_generate_output=True, output_hidden_states=True, max_new_tokens=80
            )
            prediction = PredictedCompletion()
            prediction.context_pos = len(self.sequence)
            completion_length = stopping_criteria[0].completion_length
            completion_end_pos = input_ids.shape[-1] + completion_length
            prediction.completion = self.resources.tokenizer.decode(outputs.sequences[0, input_ids.shape[-1]:completion_end_pos], skip_special_tokens=False)
            prediction.response = self.resources.tokenizer.decode(outputs.sequences[0, completion_end_pos:], skip_special_tokens=False).rstrip(turn_switch_token)
            response_speaker_match = re.search(self.any_identity_regex, prediction.response)
            if response_speaker_match:
                prediction.response = prediction.response[response_speaker_match.end()+1:]
                prediction.response_speaker = response_speaker_match[0]
            is_turn_switch = True
            if completion_length > 0:
                prediction.embedding = torch.cat([pos_states[-1] for pos_states in outputs.hidden_states[1:completion_length+1]], dim=1).mean(dim=1)
                is_turn_switch = False

        return prediction, is_turn_switch, similarity_with_last_pred

    def _cache_response(self, response):
        #print(f"_cache_response called: '{response}'")
        self.response_cache.clear()
        # divide response into k chunks to be released one at a time at every agent interval
        response_tokens = self.resources.tokenizer.encode(response, add_special_tokens=False)
        chunk_size = self.generate_kwargs["max_new_tokens"]
        num_chunks = math.ceil(len(response_tokens) / chunk_size)
        for i in range(num_chunks):
            chunk = self.resources.tokenizer.decode(response_tokens[i*chunk_size:(i+1)*chunk_size], skip_special_tokens=False)
            self.response_cache.append(chunk)

    def _release_cached_response_chunk(self):
        if len(self.response_cache) > 0:
            released_chunk = self.response_cache.popleft()
            #print (f"_release_cached_response_chunk called: '{released_chunk}'")
            return released_chunk
        #print ("_release_cached_response_chunk called: None")
        return None

    def reset(self):
        self.sequence = ""
        self.partial_pos = -1
        self._set_prefix()
        self._set_current_speaker(self.config.user_identity)
        self.last_cycle_time = datetime.now()
        self.last_user_pause_time = None
        self.agent_pause_duration = 0.0
        self.response_cache = deque()

    def set_config(self, config):
        do_set_prefix = config.identities != self.config.identities
        self.config = config
        if do_set_prefix:
            self._set_prefix()

    def execute(self, next_input=None):
        output = None
        output_for_cache = None
        sequence_changed = False

        # Check for new input:
        sequence_changed = self._update_sequence_from_input(next_input)

        # If no new input and the user is currently speaking, append an incrementing pause for the user:
        if not sequence_changed and self.current_speaker == self.config.user_identity:
            if self.last_user_pause_time is None:
                self.last_user_pause_time = datetime.now()
            seconds_since_last_user_pause = (datetime.now() - self.last_user_pause_time).total_seconds()
            if seconds_since_last_user_pause >= 0.1:
                user_pause = self._incrementing_pause(seconds_since_last_user_pause)
                self.sequence += user_pause
                sequence_changed = True
                self.last_user_pause_time = datetime.now()
        else:
            self.last_user_pause_time = None

        # If it is not time to run the next predict/output cycle yet, just return nothing.
        # Otherwise, set the last cycle time to now and proceed.
        seconds_since_last_cycle = (datetime.now() - self.last_cycle_time).total_seconds()
        if seconds_since_last_cycle < max(self.config.interval, self.agent_pause_duration):
            return output, output_for_cache, sequence_changed
        self.last_cycle_time = datetime.now()

        # Predict continuation
        self.agent_pause_duration = 0.0
        self._trim_sequence()

        release_prediction = True
        num_cached_chunks_released = 0
        if self.config.predictive_lookahead and self.current_speaker != self.config.agent_identity:
            self.last_prediction, is_turn_switch, similarity_with_last_pred = self._predict_completion(self.last_prediction)
            is_predicted_agent_response = self.last_prediction.response_speaker == self.config.agent_identity
            release_prediction = False
            if similarity_with_last_pred < self.config.similarity_threshold:
                self._cache_response(self.last_prediction.response)
                # only release the predicted response for downstream caching if it is an agent response
                if is_predicted_agent_response:
                    # ~ indicates a response candidate to be chunked, pre-cached, and released later 
                    # by downstream processors (e.g., TTS handler)
                    output_for_cache = f"~[{len(self.response_cache)}]{self.last_prediction.response}"
            if is_turn_switch and is_predicted_agent_response:
                self._set_current_speaker(self.config.agent_identity)
                release_prediction = True
        
        prediction = ""
        if release_prediction:
            while not prediction or re.search(self.incomplete_pause_regex, prediction):
                cached_chunk = self._release_cached_response_chunk()
                if cached_chunk is not None:
                    num_cached_chunks_released += 1
                    prediction += cached_chunk
                else:
                    num_cached_chunks_released = 0
                    stopping_criteria = self.pause_regex if not prediction else self.end_pause_regex
                    prediction += self._generate(f"{self.sequence}{prediction}", stopping_criteria=stopping_criteria)
        prediction_lstrip = prediction.lstrip()

        # If prediction is a turn switch to agent, switch to the agent and output the prediction:
        if prediction_lstrip.startswith(self.config.agent_identity):
            output = re.search(self.agent_turn_switch_regex, prediction)[0]
            self._set_current_speaker(self.config.agent_identity)

        # If prediction is a turn switch to user and the agent is currently speaking, append and output an 
        # incrementing pause for the agent:
        elif self.current_speaker == self.config.agent_identity and prediction_lstrip.startswith(self.config.user_identity):
            output = self._incrementing_pause(seconds_since_last_cycle)
            # since the agent pauses after every execute cycle, the actual pause duration should remain constant
            # even though it is incrementing on the sequence.
            self.agent_pause_duration = self.config.interval

        # If prediction is not a turn switch and the agent is currently speaking, output the prediction,
        # otherwise suppress the prediction (output nothing):
        elif self.current_speaker == self.config.agent_identity:
            agent_continuation = re.search(self.speaker_continue_regex, prediction)
            if agent_continuation:
                output = agent_continuation[0]

        if output:
            # suppress anything that comes after a turn switch prediction (including an incomplete one at the end).
            # turn switch predictions must be the first thing in the prediction in order to be processed.
            identity_match = re.search(self.any_identity_with_incomplete_regex, output)
            if identity_match:
                output = output[:identity_match.start()]
            self.sequence += output
            sequence_changed = True
            # if the agent pause duration hasn't been explicitly set, try to locate it in the output.
            if not self.agent_pause_duration > 0.0:
                agent_pause = re.search(self.pause_regex, output)
                if agent_pause:
                    self.agent_pause_duration = self.config.interval if agent_pause[1] == "." else float(agent_pause[1])
                    self.agent_pause_duration = min(self.agent_pause_duration, self.config.max_agent_pause_duration)
            # Otherwise, it has been explicitly set by an incrementing pause, so override the output with the actual pause value.
            # this ensures downstream processors (e.g., TTS handler) get the correct pause value, even though the sequence will
            # contain the cumulative incremental pause value.
            else:
                pause_prefix = "<p> " if self.config.add_special_pause_token else ""
                output = f" {pause_prefix}({self.agent_pause_duration:.1f})"

            if num_cached_chunks_released > 0:
                # * indicates to downstream processors (e.g., TTS handler) that this output represents a cached response chunk
                # that should be released from cache instead of rendered from scratch
                output = f"*[{num_cached_chunks_released}]{output}"

        return output, output_for_cache, sequence_changed

class RealtimeAgentMultiprocessing:
    def __init__(self, wait_until_running=True, config=None, modelpath="AbrahamSanders/opt-2.7b-realtime-chat-v2",
                 device=None, chain_to_input_queue=None, output_sequence=False, output_sequence_max_length=None):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.reset_queue = ctx.Queue()
        self.config_queue = ctx.SimpleQueue()
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.sequence_queue = ctx.Queue()
        self.chain_to_input_queue = chain_to_input_queue
        self.output_sequence = output_sequence
        self.output_sequence_max_length = output_sequence_max_length
        self.running = ctx.Value(c_bool, False)

        self.execute_process = ctx.Process(target=self.execute, daemon=True, args=(config, modelpath, device))
        self.execute_process.start()

        if wait_until_running:
            self.wait_until_running()

    def wait_until_running(self):
        #TODO: use an Event instead of a loop
        while not self.is_running():
            sleep(0.01)

    def is_running(self):
        return self.running.value

    def execute(self, config, modelpath, device):
        agent_resources = RealtimeAgent_Resources(modelpath=modelpath, device=device)
        agent = RealtimeAgent(resources=agent_resources, config=config)

        last_speaker = agent.current_speaker
        self.running.value = True
        while True:
            try:
                new_config = queue_helpers.skip_queue(self.config_queue)
                if new_config is not None:
                    config = new_config
                    agent.set_config(config)

                if queue_helpers.skip_queue(self.reset_queue) is not None:
                    agent.reset()

                next_input = queue_helpers.join_queue(self.input_queue)
                output, output_for_cache, sequence_changed = agent.execute(next_input)
                
                if output_for_cache and self.chain_to_input_queue is not None:
                    self.chain_to_input_queue.put(output_for_cache)
                
                if output:
                    self.output_queue.put(output)
                    if self.chain_to_input_queue is not None:
                        self.chain_to_input_queue.put(output)

                #HACK: if the user has started speaking, send an empty pause to the chained input queue
                #      (usually the TTS handler) to force any agent utterances stuck in the buffer to be processed.
                if self.chain_to_input_queue is not None and agent.current_speaker == agent.config.user_identity \
                                                         and last_speaker != agent.config.user_identity:
                    self.chain_to_input_queue.put(" (0.0)")
                last_speaker = agent.current_speaker

                if self.output_sequence and sequence_changed:
                    max_length = 0 if self.output_sequence_max_length is None else self.output_sequence_max_length
                    self.sequence_queue.put(agent.sequence[-max_length:])
            except Exception as ex:
                #TODO: logging here
                print(ex)
                #raise ex
            sleep(0.05)

    def queue_config(self, config):
        self.config_queue.put(config)

    def queue_reset(self):
        self.reset_queue.put(True)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        return queue_helpers.join_queue(self.output_queue, delim="")

    def next_sequence(self):
        return queue_helpers.skip_queue(self.sequence_queue)
