from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
import torch
import re
from threading import Thread, Lock
from queue import SimpleQueue
from time import sleep

from .identity import Identity
from .utils import queue_helpers

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

    def create_agent(self, **kwargs):
        return RealtimeAgent(resources=self, **kwargs)

class RealtimeAgent:
    def __init__(self, resources=None, identities=None, user_identity="S1", agent_identity="S2", 
                 interval=0.5, max_history_words=250, max_agent_pause_duration=10.0, random_state=None,
                 chain_to_input_queue=None):
        if resources is None:
            resources = RealtimeAgent_Resources()
        self.resources = resources

        if identities is None:
            identities = Identity.default_identities()
        self.identities = identities
        self.user_identity = user_identity
        self.agent_identity = agent_identity
        self.interval = interval
        self.max_history_words = max_history_words
        self.max_agent_pause_duration = max_agent_pause_duration
        self.random_state = random_state

        self.tokenizer_max_length = None
        if (not self.resources.tokenizer.model_max_length or self.resources.tokenizer.model_max_length > 9999) \
              and hasattr(self.resources.model.config, "max_position_embeddings"):
            self.tokenizer_max_length = self.resources.model.config.max_position_embeddings

        self.generate_kwargs = {
            "pad_token_id": self.resources.tokenizer.pad_token_id,
            "eos_token_id": self.resources.tokenizer.eos_token_id,
            "max_new_tokens": 5,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50,
            "temperature": 1.2,
            #"num_beams": 2,
            #"early_stopping": True
        }

        self.any_identity_regex = re.compile(r"S\d+?")
        self.any_identity_with_incomplete_regex = re.compile(rf" (?:{self.any_identity_regex.pattern}|S\Z)")
        self.agent_turn_switch_regex = re.compile(rf"(?<={self.agent_identity}:).+?(?= {self.any_identity_regex.pattern}|\Z)")
        self.agent_continue_regex = re.compile(rf".+?(?= {self.any_identity_regex.pattern}|\Z)")
        self.pause_regex = re.compile(r"\(\d*?\.\d*?\)")
        self.pause_at_end_regex = re.compile(rf"{self.pause_regex.pattern}\Z")
        self.incomplete_pause_regex = re.compile(r"\(\d*?\.?\d*?\Z")
        self.end_pause_regex = re.compile(r"\)")
        
        self.input_queue = SimpleQueue()
        self.output_queue = SimpleQueue()
        self.chain_to_input_queue = chain_to_input_queue
        self.execute_lock = Lock()

        self.reset()
        self.execute_thread = Thread(target=self.execute, daemon=True)
        self.execute_thread.start()

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
        if self.random_state is not None:
            set_seed(self.random_state)

        generate_result = self.resources.model.generate(
            input_ids=inputs.input_ids,
            attention_mask = inputs.attention_mask,
            return_dict_in_generate=True,
            **generate_kwargs
        )
        result_ids = generate_result.sequences.cpu()

        # Decode and return results
        results = []
        for i in range(result_ids.shape[0]):
            response_start_idx = inputs.input_ids.shape[-1]
            generated_text = self.resources.tokenizer.decode(result_ids[i, response_start_idx:], 
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
        if len(history_split) > self.max_history_words:
            history_split = history_split[-self.max_history_words:]
            for i in range(len(history_split)):
                if re.match(self.any_identity_regex, history_split[i]):
                    break
            i = 0 if i == len(history_split)-1 else i
            trimmed_history = " ".join(history_split[i:])
            self.sequence = f"{self.sequence[:self.prefix_length]}{trimmed_history}"

    def _incrementing_pause(self):
        # get previous pause duration (if any)
        pause_match = re.search(self.pause_at_end_regex, self.sequence[-10:])
        if pause_match:
            pause_duration = self.interval if pause_match[0] == "(.)" else float(pause_match[0][1:-1])
            pause_match_len = pause_match.end()-pause_match.start()+1
            self.sequence = self.sequence[:-pause_match_len]
        else:
            pause_duration = 0.0

        # increment and return
        pause_duration += self.interval
        return f" ({pause_duration:.1f})"
        

    def reset(self):
        with self.execute_lock:
            self.sequence = ""
            for identity, info in self.identities.items():
                self.sequence += f"<participant> {identity} (name: {info.name}, age: {info.age}, sex: {info.sex}) "
            self.sequence += "<dialog>"
            self.prefix_length = len(self.sequence)+1
            self._set_current_speaker(self.user_identity)

    def execute(self):
        while True:
            agent_pause_duration = 0.0
            with self.execute_lock:
                try:
                    # Check for new input:
                    next_input = queue_helpers.join_queue(self.input_queue)
                    if next_input:
                        if self.current_speaker != self.user_identity:
                            self._set_current_speaker(self.user_identity)
                        self.sequence += f" {next_input}"
                    # If no new input and the user is currently speaking, append an incrementing pause for the user:
                    elif self.current_speaker == self.user_identity:
                        user_pause = self._incrementing_pause()
                        self.sequence += user_pause

                    # Predict continuation
                    self._trim_sequence()
                    prediction = ""
                    while not prediction or re.search(self.incomplete_pause_regex, prediction):
                        stopping_criteria = self.pause_regex if not prediction else self.end_pause_regex
                        prediction += self._generate(f"{self.sequence}{prediction}", stopping_criteria=stopping_criteria)
                    prediction_lstrip = prediction.lstrip()
                    output = None

                    # If prediction is a turn switch to agent, switch to the agent and output the prediction:
                    if prediction_lstrip.startswith(self.agent_identity):
                        output = re.search(self.agent_turn_switch_regex, prediction)[0]
                        self._set_current_speaker(self.agent_identity)

                    # If prediction is a turn switch to user and the agent is currently speaking, append and output an 
                    # incrementing pause for the agent:
                    elif self.current_speaker == self.agent_identity and prediction_lstrip.startswith(self.user_identity):
                        output = self._incrementing_pause()
                        # since the agent pauses after every execute cycle, the actual pause duration should remain constant
                        # even though it is incrementing on the sequence.
                        agent_pause_duration = self.interval

                    # If prediction is not a turn switch and the agent is currently speaking, output the prediction,
                    # otherwise suppress the prediction (output nothing):
                    elif self.current_speaker == self.agent_identity:
                        output = re.search(self.agent_continue_regex, prediction)[0]

                    if output:
                        # suppress anything that comes after a turn switch prediction (including an incomplete one at the end).
                        # turn switch predictions must be the first thing in the prediction in order to be processed.
                        identity_match = re.search(self.any_identity_with_incomplete_regex, output)
                        if identity_match:
                            output = output[:identity_match.start()]
                        self.sequence += output
                        self.output_queue.put(output)
                        if self.chain_to_input_queue is not None:
                            self.chain_to_input_queue.put(output)
                        # if the agent pause duration hasn't been explicitly set, try to locate it in the output.
                        if not agent_pause_duration > 0.0:
                            agent_pause = re.search(self.pause_regex, output)
                            if agent_pause:
                                agent_pause_duration = self.interval if agent_pause[0] == "(.)" else float(agent_pause[0][1:-1])
                                agent_pause_duration = min(agent_pause_duration, self.max_agent_pause_duration)
                except:
                    #TODO: logging here
                    pass
                
            sleep(self.interval + agent_pause_duration)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        return queue_helpers.join_queue(self.output_queue, delim="")

class RealtimeAgentMultiprocessing:
    def __init__(self, wait_until_running=True, user_identity="S1", agent_identity="S2", **kwargs):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.reset_queue = ctx.SimpleQueue()
        self.input_queue = ctx.SimpleQueue()
        self.output_queue = ctx.SimpleQueue()
        self.running = ctx.Value(c_bool, False)
        # Needed because these properties should be externally visible.
        # TODO: Find a better way to do this.
        self.user_identity = user_identity
        self.agent_identity = agent_identity
        kwargs["user_identity"] = user_identity
        kwargs["agent_identity"] = agent_identity

        self.execute_process = ctx.Process(target=self.execute, daemon=True, kwargs=kwargs)
        self.execute_process.start()

        if wait_until_running:
            #TODO: use an Event instead of a loop
            while not self.is_running():
                sleep(0.01)


    def execute(self, **kwargs):
        agent = RealtimeAgent(**kwargs)
        self.running.value = True
        while True:
            if not self.reset_queue.empty():
                self.reset_queue.get()
                agent.reset()
            queue_helpers.transfer_queue(self.input_queue, agent.input_queue)
            queue_helpers.transfer_queue(agent.output_queue, self.output_queue)
            sleep(0.01)

    def reset(self):
        self.reset_queue.put(True)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        return queue_helpers.join_queue(self.output_queue, delim="")

    def is_running(self):
        return self.running.value
