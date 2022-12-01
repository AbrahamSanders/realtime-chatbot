import whisper
import numpy as np
from whisper import tokenizer, _MODELS
from whisper.audio import SAMPLE_RATE
from time import sleep
from collections import deque

from .utils import queue_helpers
from .utils import audio_helpers

class ASRConfig:
    def __init__(self, model_size = 'medium.en', lang="English", n_context_segs=1, 
                 logprob_threshold=-0.4, no_speech_threshold=0.3, buffer_size=3):
        self.model_size = model_size
        self.lang = lang
        self.n_context_segs = n_context_segs
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.buffer_size = buffer_size

class ASRHandlerMultiprocessing:
    def __init__(self, wait_until_running=True, config=None, chain_to_input_queue=None):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.SimpleQueue()
        self.output_queue = ctx.SimpleQueue()
        self.config_queue = ctx.SimpleQueue()
        self.chain_to_input_queue = chain_to_input_queue
        self.running = ctx.Value(c_bool, False)

        self.AUTO_DETECT_LANG = "Auto Detect"
        self.available_languages = sorted(tokenizer.TO_LANGUAGE_CODE.keys())
        self.available_languages = [lang.capitalize() for lang in self.available_languages]
        self.available_languages = [self.AUTO_DETECT_LANG] + self.available_languages
        self.available_model_sizes = list(_MODELS)

        self.execute_process = ctx.Process(target=self.execute, daemon=True, args=(config,))
        self.execute_process.start()

        if wait_until_running:
            #TODO: use an Event instead of a loop
            while not self.is_running():
                sleep(0.01)

    def execute(self, config):
        if config is None:
            config = ASRConfig()
        model = whisper.load_model(config.model_size)
        last_n_segs = deque(maxlen=config.n_context_segs)
        input_buffer = []

        self.running.value = True
        while True:
            try:
                # If new config available, reconfigure the model and context queue and
                # replace the old config
                if not self.config_queue.empty():
                    new_config = self.config_queue.get()
                    if new_config.model_size != config.model_size:
                        model = whisper.load_model(new_config.model_size)
                    if new_config.n_context_segs != config.n_context_segs:
                        last_n_segs = deque(maxlen=new_config.n_context_segs)
                    config = new_config

                # If audio input is available, process it and output the resulting text (if any)
                if not self.input_queue.empty():
                    input_buffer.append(self.input_queue.get())
                
                if len(input_buffer) >= config.buffer_size:
                    next_input = (input_buffer[0][0], np.concatenate([buf[1] for buf in input_buffer]))
                    input_buffer.clear()
                    audio = audio_helpers.convert_sample_rate(next_input[1], next_input[0], SAMPLE_RATE)

                    initial_prompt = " ".join([seg for seg in last_n_segs if seg])
                    transcription = model.transcribe(
                        audio,
                        language = config.lang if config.lang != self.AUTO_DETECT_LANG else None,
                        logprob_threshold=config.logprob_threshold,
                        no_speech_threshold=config.no_speech_threshold,
                        initial_prompt = initial_prompt
                    )
                    transcription_text = transcription['text'].strip()
                    if len(last_n_segs) == last_n_segs.maxlen:
                        last_n_segs.popleft()
                    last_n_segs.append(transcription_text)
                    
                    if transcription_text:
                        self.output_queue.put(transcription_text)
                        if self.chain_to_input_queue is not None:
                            self.chain_to_input_queue.put(transcription_text)
            except:
                #TODO: logging here
                pass
            sleep(0.01)

    def queue_config(self, config):
        self.config_queue.put(config)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        return queue_helpers.join_queue(self.output_queue)

    def is_running(self):
        return self.running.value
