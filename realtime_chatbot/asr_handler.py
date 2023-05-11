import whisper
from whisper import tokenizer, _MODELS
from whisper.audio import SAMPLE_RATE
from time import sleep
from collections import deque

from .utils import queue_helpers
from .utils import audio_helpers

class ASRConfig:
    def __init__(self, model_size = 'base.en', lang="English", n_context_segs=1, n_prefix_segs=1, 
                 compression_ratio_threshold=2.4, logprob_threshold=-0.7, no_speech_threshold=0.6, 
                 max_buffer_size=5):
        self.model_size = model_size
        self.lang = lang
        self.n_context_segs = n_context_segs
        self.n_prefix_segs = n_prefix_segs
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.max_buffer_size = max_buffer_size

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class ASRHandlerMultiprocessing:
    def __init__(self, wait_until_running=True, config=None, device=None, chain_to_input_queue=None, 
                 output_debug_audio=False):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.config_queue = ctx.SimpleQueue()
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.chain_to_input_queue = chain_to_input_queue
        if output_debug_audio:
            self.debug_audio_queue = ctx.Queue()
        self.output_debug_audio = output_debug_audio
        self.running = ctx.Value(c_bool, False)

        self.AUTO_DETECT_LANG = "Auto Detect"
        self.available_languages = sorted(tokenizer.TO_LANGUAGE_CODE.keys())
        self.available_languages = [lang.capitalize() for lang in self.available_languages]
        self.available_languages = [self.AUTO_DETECT_LANG] + self.available_languages
        self.available_model_sizes = list(_MODELS)

        self.execute_process = ctx.Process(target=self.execute, daemon=True, args=(config, device))
        self.execute_process.start()

        if wait_until_running:
            self.wait_until_running()

    def wait_until_running(self):
        #TODO: use an Event instead of a loop
        while not self.is_running():
            sleep(0.01)

    def is_running(self):
        return self.running.value

    def _get_prompt_and_prefix(self, config, last_n_segs):
        prompt = prefix = ""
        prefix_audio = []
        prefix_count = 0
        if last_n_segs:
            for i in range(len(last_n_segs)-1, -1, -1):
                seg_audio, seg_text = last_n_segs[i]
                if prefix_count < config.n_prefix_segs:
                    prefix_audio.append(seg_audio)
                    if seg_text:
                        prefix = f"{seg_text} {prefix}"
                    prefix_count += 1
                elif seg_text:
                    prompt = f"{seg_text} {prompt}"
            prompt = prompt.rstrip()
            prefix = prefix.rstrip()
            prefix_audio.reverse()
        return prompt, prefix, prefix_audio

    def _whisper_transcribe(self, model, config, next_input, cached_resample, last_n_segs=None):
        initial_prompt, prefix, prefix_audio = self._get_prompt_and_prefix(config, last_n_segs)
        #print(f"prompt: '{initial_prompt}'; prefix: '{prefix}'")
        if prefix_audio:
            prefixed_next_input = audio_helpers.concat_audios_to_tensor(prefix_audio + [next_input])
        else:
            prefixed_next_input = next_input

        audio = prefixed_next_input[0].to(device=model.device)
        audio, cached_resample = audio_helpers.downsample(audio, prefixed_next_input[1], SAMPLE_RATE, cached_resample)
        if self.output_debug_audio:
            self.debug_audio_queue.put((SAMPLE_RATE, audio.cpu().numpy()))

        transcription = model.transcribe(
            audio,
            initial_prompt = initial_prompt, 
            prefix = prefix,
            language = config.lang if config.lang != self.AUTO_DETECT_LANG else None,
            compression_ratio_threshold = config.compression_ratio_threshold,
            logprob_threshold = config.logprob_threshold,
            no_speech_threshold = config.no_speech_threshold,
            without_timestamps = True,
            beam_size = 2,
            best_of = 2,
            sample_len = 15,
            temperature = 0.0
        )
        transcription_text = transcription['text'].replace("...", "").strip()

        # If the model is repeating itself, try again with just the original audio (no prompt or prefix).
        # Otherwise, the repetition will make it into the next prefix and the model will descend into an
        # endless positive feedback loop of repetition.
        if transcription_text and (initial_prompt or prefix) and (
                    transcription_text in prefix or transcription_text in initial_prompt or
                    transcription['segments'][0]["compression_ratio"] > config.compression_ratio_threshold):
            #print("Trying again with just original audio...")
            transcription_text, transcription, cached_resample = self._whisper_transcribe(
                model, config, next_input, cached_resample
            )
                    
        return transcription_text, transcription, cached_resample

    def _output_transcription(self, transcription_text, is_partial=False, output_blank=False):
        if transcription_text or output_blank:
            prefix = "~" if is_partial else "*"
            transcription_text = f"{prefix}{transcription_text}"
            self.output_queue.put(transcription_text)
            if self.chain_to_input_queue is not None:
                self.chain_to_input_queue.put(transcription_text)

    def execute(self, config, device):
        if config is None:
            config = ASRConfig()
        model = whisper.load_model(config.model_size, device=device)
        last_n_segs = deque(maxlen=config.n_context_segs+config.n_prefix_segs)
        cached_resample = None
        input_buffer = []
        buffer_size = 1
        ends_with_silence_count = 0

        self.running.value = True
        while True:
            try:
                # If new config available, reconfigure the model and context queue and
                # replace the old config
                new_config = queue_helpers.skip_queue(self.config_queue)
                if new_config is not None:
                    if new_config.model_size != config.model_size:
                        print(f"Loading model {new_config.model_size}...")
                        model = whisper.load_model(new_config.model_size, device=device)
                    if (new_config.n_context_segs != config.n_context_segs 
                            or new_config.n_prefix_segs != config.n_prefix_segs):
                        last_n_segs = deque(maxlen=new_config.n_context_segs+new_config.n_prefix_segs)
                    config = new_config

                # If audio input is available, buffer it up
                queue_helpers.transfer_queue_to_buffer(self.input_queue, input_buffer)
                
                if len(input_buffer) >= buffer_size:
                    # First, check the if the last received segment ends with silence. Otherwise,
                    # if buffer_size < config.max_buffer_size, we can extend the buffer and collect more 
                    # audio before outputting a final transcription for the full contents of the buffer. 
                    next_input_last_fragment = audio_helpers.concat_audios_to_tensor(input_buffer[-1:])
                    # don't pass last_n_segs because we don't want silence detection to be conditioned
                    # on prior context - doing so could induce hallucinated speech.
                    transcription_text, _, cached_resample = self._whisper_transcribe(
                        model, config, next_input_last_fragment, cached_resample
                    )
                    if transcription_text:
                        ends_with_silence_count = 0
                    else:
                        ends_with_silence_count += 1

                    # Next, transcribe the entire contents of the input buffer with sliding window context (last_n_segs). 
                    # If buffer_size is 1 and we already have a detected silence, skip this step and output blank
                    # since attempting to transcribe a short silence in context often leads to hallucination.
                    next_input = audio_helpers.concat_audios_to_tensor(input_buffer)
                    if buffer_size == 1 and ends_with_silence_count > 0:
                        transcription_text = ""
                    else:
                        transcription_text, _, cached_resample = self._whisper_transcribe(
                            model, config, next_input, cached_resample, last_n_segs
                        )

                    # Next, determine if we should keep waiting for more audio and output a partial transcription or 
                    # clear the buffer and finalize the transcription. If speech is still streaming in, 
                    # wait for more audio as long as there is room to extend the buffer.
                    if buffer_size < config.max_buffer_size and ends_with_silence_count < 1:
                        # output a partial transcription of the buffered audio segment and then extend the buffer
                        self._output_transcription(transcription_text, is_partial=True)
                        buffer_size += 1
                    else:
                        # output the final transcription of the buffered audio segment and then reset the buffer
                        output_blank = buffer_size > 1
                        self._output_transcription(transcription_text, output_blank=output_blank)

                        input_buffer.clear()
                        buffer_size = 1
                        ends_with_silence_count = 0
                        if len(last_n_segs) == last_n_segs.maxlen:
                            last_n_segs.popleft()
                        last_n_segs.append((next_input, transcription_text))
            except Exception as ex:
                #TODO: logging here
                print(ex)
                #raise ex
            sleep(0.05)

    def queue_config(self, config):
        self.config_queue.put(config)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        return queue_helpers.join_queue(self.output_queue)

    def next_debug_audio(self):
        return queue_helpers.join_queue_audio(self.debug_audio_queue)
        