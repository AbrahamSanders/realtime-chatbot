from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
import torch
import re
import uuid
from time import sleep
from datetime import datetime

from .identity import Identity
from .utils import queue_helpers

class RealtimeAgent_Resources:
    def __init__(self, modelpath="AbrahamSanders/opt-2.7b-realtime-chat", device=None):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)
        self.tokenizer.truncation_side = "left"
        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(modelpath)
        self.model = self.model.to(self.device)

    def create_agent(self, config=None):
        return RealtimeAgent(resources=self, config=config)

class RealtimeAgentConfig:
    def __init__(self, identities=None, user_identity="S1", agent_identity="S2", 
                 interval=0.4, max_history_words=100, max_agent_pause_duration=10.0, random_state=None):
        if identities is None:
            identities = Identity.default_identities()
        self.identities = identities
        self.user_identity = user_identity
        self.agent_identity = agent_identity
        self.interval = interval
        self.max_history_words = max_history_words
        self.max_agent_pause_duration = max_agent_pause_duration
        self.random_state = random_state

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

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
            "max_new_tokens": 10,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 70,
            "temperature": 1.8,
            "num_beams": 2,
            "early_stopping": True
        }

        self.any_identity_regex = re.compile(r"S\d+?")
        self.any_identity_with_incomplete_regex = re.compile(rf" (?:{self.any_identity_regex.pattern}|S\Z)")
        self.agent_turn_switch_regex = re.compile(rf"(?<={self.config.agent_identity}:).+?(?= {self.any_identity_regex.pattern}|\Z)")
        self.speaker_continue_regex = re.compile(rf".+?(?= {self.any_identity_regex.pattern}|\Z)")
        self.pause_regex = re.compile(r"\(\d*?\.\d*?\)")
        self.pause_at_end_regex = re.compile(rf"{self.pause_regex.pattern}\Z")
        self.incomplete_pause_regex = re.compile(r"\(\d*?\.?\d*?\Z")
        self.end_pause_regex = re.compile(r"\)")
        self.input_segments_regex = re.compile(" (?=[~*])")
        self.sequence_split_regex = re.compile(rf"\s(?={self.any_identity_regex.pattern})")

        self.prefix_length = None
        
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
        if self.config.random_state is not None:
            set_seed(self.config.random_state)

        generate_result = self.resources.model.generate(
            input_ids=inputs.input_ids,
            attention_mask = inputs.attention_mask,
            return_dict_in_generate=True,
            **generate_kwargs
        )
        result_ids = generate_result.sequences.cpu()

        # Decode and return results
        results = []
        take_tokens = 5
        for i in range(result_ids.shape[0]):
            response_start_idx = inputs.input_ids.shape[-1]
            response_end_idx = response_start_idx + take_tokens
            generated_text = self.resources.tokenizer.decode(result_ids[i, response_start_idx:response_end_idx], 
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

        gen_results = results if len(results) > 1 else results[0]
        return gen_results

    def _set_current_speaker(self, identity):
        self.sequence += f" {identity}:"
        self.current_speaker = identity

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

    def _incrementing_pause(self, seconds_since_last_cycle):
        # get previous pause duration (if any)
        pause_match = re.search(self.pause_at_end_regex, self.sequence[-10:])
        if pause_match:
            pause_duration = self.config.interval if pause_match[0] == "(.)" else float(pause_match[0][1:-1])
            pause_match_len = pause_match.end()-pause_match.start()+1
            self.sequence = self.sequence[:-pause_match_len]
        else:
            pause_duration = 0.0

        # increment and return
        pause_duration += seconds_since_last_cycle
        return f" ({pause_duration:.1f})"
        
    def _set_prefix(self):
        prefix = ""
        for identity, info in self.config.identities.items():
            prefix += f"<participant> {identity} (name: {info.name}, age: {info.age}, sex: {info.sex}) "
        prefix += "<dialog>"
        prev_prefix_length = None
        if len(self.sequence) > 0 and self.prefix_length is not None:
            self.sequence = f"{prefix} {self.sequence[self.prefix_length:]}"
            prev_prefix_length = self.prefix_length
        else:
            self.sequence = prefix
        self.prefix_length = len(prefix)+1
        if prev_prefix_length is not None and self.partial_pos > -1:
            self.partial_pos += self.prefix_length-prev_prefix_length

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
                for utt in utterances_after_partial_pos:
                    if utt:
                        identity_match = re.match(self.any_identity_regex, utt)
                        if identity_match and not utt.startswith(self.config.user_identity):
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
                            if has_user_identity:
                                self._set_current_speaker(self.config.user_identity)
                            self.sequence += f" {seg_text[:next_slice_idx]}"
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

    def reset(self):
        self.sequence = ""
        self.partial_pos = -1
        self._set_prefix()
        self._set_current_speaker(self.config.user_identity)
        self.last_cycle_time = datetime.now()
        self.agent_pause_duration = 0.0

    def set_config(self, config):
        do_set_prefix = config.identities != self.config.identities
        self.config = config
        if do_set_prefix:
            self._set_prefix()

    def execute(self, next_input=None):
        output = None
        sequence_changed = False

        # Check for new input:
        sequence_changed = self._update_sequence_from_input(next_input)

        # If it is not time to run the next predict/output cycle yet, just return nothing.
        # Otherwise, set the last cycle time to now and proceed.
        seconds_since_last_cycle = (datetime.now() - self.last_cycle_time).total_seconds()
        if seconds_since_last_cycle < max(self.config.interval, self.agent_pause_duration):
            return output, sequence_changed
        self.last_cycle_time = datetime.now()

        # If no new input and the user is currently speaking, append an incrementing pause for the user:
        if not sequence_changed and self.current_speaker == self.config.user_identity:
            user_pause = self._incrementing_pause(seconds_since_last_cycle)
            self.sequence += user_pause
            sequence_changed = True

        # Predict continuation
        self.agent_pause_duration = 0.0
        self._trim_sequence()
        prediction = ""
        while not prediction or re.search(self.incomplete_pause_regex, prediction):
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
            output = re.search(self.speaker_continue_regex, prediction)[0]

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
                    self.agent_pause_duration = self.config.interval if agent_pause[0] == "(.)" else float(agent_pause[0][1:-1])
                    self.agent_pause_duration = min(self.agent_pause_duration, self.config.max_agent_pause_duration)

        #print (f"Agent loop done: {str(uuid.uuid4())[:8]}")
        return output, sequence_changed

class RealtimeAgentMultiprocessing:
    def __init__(self, wait_until_running=True, config=None, modelpath="AbrahamSanders/opt-2.7b-realtime-chat",
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
                output, sequence_changed = agent.execute(next_input)
                if output:
                    self.output_queue.put(output)
                    if self.chain_to_input_queue is not None:
                        self.chain_to_input_queue.put(output)

                if self.output_sequence and sequence_changed:
                    max_length = 0 if self.output_sequence_max_length is None else self.output_sequence_max_length
                    self.sequence_queue.put(agent.sequence[-max_length:])
            except:
                #TODO: logging here
                pass
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
