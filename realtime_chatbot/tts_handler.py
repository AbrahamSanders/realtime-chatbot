import torch
import re
import abc
import math
from time import sleep
from collections import deque
import os

from .utils import queue_helpers, audio_helpers
from .speech_enhancer import SpeechEnhancer

class TTSConfig:
    def __init__(self, tts_engine="fastspeech2", buffer_size=4, downsampling_factor=1, speaker=None, enhancement_model="none",
                 duration_factor=1.0, pitch_factor=1.0, energy_factor=1.0):
        if tts_engine not in ["fastspeech2", "bark"]:
            raise ValueError(f"tts_engine must be one of [fastspeech2, bark], but got {tts_engine}")
        if speaker is None:
            speaker = "Voice 1" if tts_engine == "fastspeech2" else "v2/en_speaker_6"
        self.tts_engine = tts_engine
        self.buffer_size = buffer_size
        self.downsampling_factor = downsampling_factor
        self.speaker = speaker
        self.enhancement_model = enhancement_model
        self.duration_factor = duration_factor
        self.pitch_factor = pitch_factor
        self.energy_factor = energy_factor

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class TTSHandler:
    def __init__(self, device=None, config=None, handle_pauses=True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if config is None:
            config = TTSConfig()
        self.config = config

        self.handle_pauses = handle_pauses

        self.pause_regex = re.compile(r"(?:<p> )?(\(\d*?\.\d*?\))")
        self.pause_value_regex = re.compile(r"\((\d*?\.\d*?)\)")
        self.cache_regex = re.compile(r"~\[(\d+)\]")
        self.cache_release_regex = re.compile(r"\*\[(\d+)\]")

        self.speech_enhancer = SpeechEnhancer(device=device)
        self.cached_resample = None
        self.audio_cache = deque()

    @abc.abstractmethod
    def _sanitize_text_for_tts(self, text):
        raise NotImplementedError()

    @abc.abstractmethod
    def _generate_audio(self, segment):
        raise NotImplementedError()
    
    def _cache_audio(self, text):
        self.audio_cache.clear()
        cache_prefix_match = re.match(self.cache_regex, text)
        if not cache_prefix_match:
            return
        num_chunks = int(cache_prefix_match[1])
        text = text[cache_prefix_match.end():]
        audio_segments = list(self.render_audio(text))
        if len(audio_segments) == 0:
            return
        wav, rate = audio_helpers.concat_audios_to_tensor(audio_segments)
        chunk_size = math.ceil(wav.shape[-1] / num_chunks)
        for i in range(num_chunks):
            chunk = wav[..., i*chunk_size:(i+1)*chunk_size]
            self.audio_cache.append((chunk, rate))

    def _release_cached_audio_chunks(self, num_chunks):
        released_chunks = []
        for _ in range(num_chunks):
            if len(self.audio_cache) > 0:
                released_chunks.append(self.audio_cache.popleft())
        return released_chunks

    def _render_segment_audio(self, segment):
        if self.handle_pauses and re.match(self.pause_regex, segment):
            # In case of a pause, convert the pause to blank audio
            pause_value = float(re.search(self.pause_value_regex, segment)[1])
            if not pause_value > 0.0:
                return None
            #print(f"TTS: Rendering pause: {segment}")
            new_rate = 22050 // self.config.downsampling_factor
            wav = torch.zeros(int(new_rate * pause_value), dtype=torch.float32, device=self.device)
        else:
            # In case of non-pause text, convert the text to speech
            segment = self._sanitize_text_for_tts(segment)
            if not segment or not re.sub("[. ]", "", segment):
                #print(f"TTS: Skipping: {segment}")
                return None
            #print(f"TTS: Rendering: {segment}")
            generated_audio = self._generate_audio(segment)
            if generated_audio is None:
                #print(f"TTS: Failed to generate audio for: {segment}")
                return None
            wav, rate = generated_audio
            # downsample & enhance (if selected)
            new_rate = rate // self.config.downsampling_factor
            wav, self.cached_resample = audio_helpers.downsample(wav, rate, new_rate, self.cached_resample)
            wav, new_rate = self.speech_enhancer.enhance(self.config.enhancement_model, wav, new_rate)
        return wav, new_rate
    
    def render_audio(self, text):
        # 1. Should the audio be cached?
        if text.startswith("~"):
            self._cache_audio(text)
            return
        # 2. Should the audio be released from the cache?
        if text.startswith("*"):
            cache_release_prefix_match = re.match(self.cache_release_regex, text)
            num_chunks = int(cache_release_prefix_match[1]) if cache_release_prefix_match else 1
            audio_segments = self._release_cached_audio_chunks(num_chunks)
            if len(audio_segments) > 0:
                for audio_segment in audio_segments:
                    yield audio_segment
                return
            else:
                text = text[cache_release_prefix_match.end():] if cache_release_prefix_match else text.lstrip("*")
        # 3. Else, render the audio and release it in chunks broken by pauses:
        #    Split the input by pauses, converting each pause to blank audio 
        #    and each non-pause to speech
        input_segs = re.split(self.pause_regex, text) if self.handle_pauses else [text]
        for input_seg in input_segs:
            audio_segment = self._render_segment_audio(input_seg)
            if audio_segment is not None:
                yield audio_segment

    def set_config(self, config):
        self.config = config
    
class FastSpeech2TTSHandler(TTSHandler):
    def __init__(self, device=None, config=None):
        super().__init__(device, config)

        from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
        from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
        import g2p_en
        from .tts_overrides import (
            get_phonemize_override, get_get_prediction_override, 
            get_generate_override, get_encoder_forward_override
        )
        g2p = g2p_en.G2p()
        TTSHubInterface.phonemize = get_phonemize_override(TTSHubInterface, g2p)
        TTSHubInterface.get_prediction = get_get_prediction_override(TTSHubInterface)
        models, cfg, self.tts_task = load_model_ensemble_and_task_from_hf_hub(
            #"facebook/fastspeech2-en-ljspeech",
            "facebook/fastspeech2-en-200_speaker-cv4",
            arg_overrides={"vocoder": "hifigan", "fp16": False}
        )
        self.tts_model = models[0].to(device)
        self.tts_model.encoder.forward = get_encoder_forward_override(self.tts_model.encoder)
        TTSHubInterface.update_cfg_with_data_cfg(cfg, self.tts_task.data_cfg)
        self.tts_generator = self.tts_task.build_generator(models, cfg)
        self.tts_generator.vocoder = self.tts_generator.vocoder.to(device)
        self.tts_generator.generate = get_generate_override(self.tts_generator)

        self.get_model_input = TTSHubInterface.get_model_input
        self.get_prediction = TTSHubInterface.get_prediction

    def _sanitize_text_for_tts(self, text):
        text = re.sub(self.pause_regex, "", text)
        text = re.sub(r"(?:\s|\A)i?[hx]+(?=(?:\s|\Z))", "", text, flags=re.IGNORECASE)
        text = re.sub(r"0 ?(?=\[)", "", text)
        text = re.sub("0[.]", "", text)
        text = re.sub(r"\[%.*?\]", "", text)
        text = re.sub(r"&=laugh.*?(?=(?:\s|\Z))", "ha! ha! ha!", text, flags=re.IGNORECASE)
        text = re.sub(r"&=.*?(?=(?:\s|\Z))", "", text)
        text = re.sub("yeah[.!?]*", "yeah,", text, flags=re.IGNORECASE)
        text = re.sub(" {2,}", " ", text)
        text = text.strip()
        return text

    def _generate_audio(self, segment):
        speaker_id = int(self.config.speaker.split()[-1])-1
        sample = self.get_model_input(self.tts_task, segment, speaker=speaker_id)
        if sample["net_input"]["src_lengths"].item() == 0:
            return None
        sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to(self.device)
        sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].to(self.device)
        if sample["speaker"] is not None:
            sample["speaker"] = sample["speaker"].to(self.device)
        
        d_factor = torch.ones_like(sample["net_input"]["src_tokens"], dtype=torch.float32)
        #d_factor[:, 1] *= config.duration_factor
        p_factor = torch.ones_like(d_factor)
        #p_factor[:, 1] *= config.pitch_factor
        wav, rate = self.get_prediction(
            self.tts_task, self.tts_model, self.tts_generator, sample, d_factor=d_factor, 
            p_factor=p_factor, e_factor=self.config.energy_factor
        )
        return wav, rate
    
class BarkTTSHandler(TTSHandler):
    def __init__(self, device=None, config=None, condition_on_previous_generation=True, reset_prev_conditioning_after=5):
        super().__init__(device, config, handle_pauses=False)

        from bark import SAMPLE_RATE
        from bark.generation import _load_history_prompt
        from .bark_api_mod import generate_audio
        from .bark_generation_mod import preload_models
        self.sample_rate = SAMPLE_RATE
        self._load_history_prompt = _load_history_prompt
        self.generate_audio = generate_audio

        preload_models(device, text_use_small=True, coarse_use_small=True, fine_use_small=True)

        self.bark_prompt_lookup = BarkTTSHandler.get_bark_prompt_lookup()
        self.speaker_history_prompt = dict(
            _load_history_prompt(self.bark_prompt_lookup[config.speaker.replace("/", os.path.sep)])
        )
        self.condition_on_previous_generation = condition_on_previous_generation
        self.reset_prev_conditioning_after = reset_prev_conditioning_after
        self.last_segment = ""
        self.last_generated_tokens = None
        self.num_consecutive_prev_conditioning = 0

    def _sanitize_text_for_tts(self, text):
        text = re.sub(self.pause_regex, "...", text)
        text = re.sub(r"(?:\s|\A)i?[hx]+(?=(?:\s|\Z))", "", text, flags=re.IGNORECASE)
        text = re.sub(r"0 ?(?=\[)", "", text)
        text = re.sub("0[.]", "", text)
        text = re.sub(r"\[% ", "[", text)
        text = re.sub(r"&=(.*?)(?=(?:\s|\Z))", lambda m: f"[{m.group(1).lower()}]", text)
        text = re.sub(r" +\.\.\.", "...", text)
        text = text.lstrip(".")
        text = re.sub(" {2,}", " ", text)
        text = text.strip()
        return text

    def _generate_audio(self, segment):
        if self.condition_on_previous_generation and self.num_consecutive_prev_conditioning < self.reset_prev_conditioning_after:
            self.num_consecutive_prev_conditioning += 1
            gen_input = " ".join([self.last_segment, segment]).lstrip()
            prefix_prompt = self.last_generated_tokens
        else:
            self.num_consecutive_prev_conditioning = 0
            gen_input = segment
            prefix_prompt = None
        
        self.last_generated_tokens, wav = self.generate_audio(
            gen_input, history_prompt=self.speaker_history_prompt, prefix_prompt=prefix_prompt,
            text_temp=0.7, waveform_temp=0.7, silent=True, output_full=True, min_eos_p=0.05, 
            max_gen_duration_s=max(1.0, len(segment.split()) / 1.25)
        )
        wav = torch.tensor((wav * 32767), dtype=torch.float32)
        rate = self.sample_rate
        self.last_segment = segment
        return wav, rate

    def set_config(self, config):
        if config.speaker != self.config.speaker:
            self.speaker_history_prompt = dict(
                self._load_history_prompt(self.bark_prompt_lookup[config.speaker.replace("/", os.path.sep)])
            )
            self.last_segment = ""
            self.last_generated_tokens = None
            self.num_consecutive_prev_conditioning = 0
        super().set_config(config)

    @staticmethod
    def get_bark_prompt_lookup():
        from bark.generation import ALLOWED_PROMPTS
        bark_prompt_lookup = {prompt: prompt for prompt in ALLOWED_PROMPTS}
        bark_prompt_lookup["unconditional"] = None
        return bark_prompt_lookup

class TTSHandlerMultiprocessing:
    def __init__(self, wait_until_running=True, config=None, device=None):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.config_queue = ctx.SimpleQueue()
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.running = ctx.Value(c_bool, False)

        if config is None or config.tts_engine == "fastspeech2":
            self.available_speakers = [f"Voice {i+1}" for i in range(200)]
        elif config.tts_engine == "bark":
            self.available_speakers = sorted(BarkTTSHandler.get_bark_prompt_lookup())

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

    def execute(self, config, device):
        if config is None:
            config = TTSConfig()

        if config.tts_engine == "fastspeech2":
            handler = FastSpeech2TTSHandler(device, config)
        elif config.tts_engine == "bark":
            handler = BarkTTSHandler(device, config)
        
        input_buffer = []
        self.running.value = True
        while True:
            try:
                new_config = queue_helpers.skip_queue(self.config_queue)
                if new_config is not None:
                    config = new_config
                    handler.set_config(config)

                queue_helpers.transfer_queue_to_buffer(self.input_queue, input_buffer)

                # First, pull any items that are slated for caching off the input buffer
                # and process them immediately.
                tmp_input_buffer = []
                for item in input_buffer:
                    # to be cached...
                    if item.startswith("~"):
                        _ = list(handler.render_audio(item))
                    # to be released from cache...
                    elif item.startswith("*"):
                        for wav, rate in handler.render_audio(item):
                            wav = wav.cpu().numpy()
                            self.output_queue.put((rate, wav))
                    else:
                        tmp_input_buffer.append(item)
                input_buffer = tmp_input_buffer

                # Next, determine what items in the input buffer to process:
                # - If the buffer is full, process the whole buffer. 
                # - If the buffer is not full but item(s) exist that contain a pause, 
                #   process everything up to and including the last pause.
                # - Otherwise, do nothing.
                items_to_process = None
                if len(input_buffer) >= config.buffer_size:
                    items_to_process = input_buffer
                elif handler.handle_pauses:
                    idx_after_last_pause = None
                    for i in range(len(input_buffer)-1, -1, -1):
                        if re.search(handler.pause_regex, input_buffer[i]):
                            idx_after_last_pause = i+1
                            break
                    if idx_after_last_pause is not None:
                        items_to_process = input_buffer[:idx_after_last_pause]
                        input_buffer = input_buffer[idx_after_last_pause:]

                if items_to_process is not None:
                    next_input = " ".join(items_to_process)
                    items_to_process.clear()
                    for wav, rate in handler.render_audio(next_input):
                        wav = wav.cpu().numpy()
                        self.output_queue.put((rate, wav))
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
        return queue_helpers.join_queue_audio(self.output_queue)
        