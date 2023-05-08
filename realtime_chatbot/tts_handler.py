import torch
import re
from time import sleep

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
            from bark.generation import ALLOWED_PROMPTS
            self.BARK_PROMPT_LOOKUP = {prompt: prompt for prompt in ALLOWED_PROMPTS}
            self.BARK_PROMPT_LOOKUP["unconditional"] = None
            self.available_speakers = sorted(self.BARK_PROMPT_LOOKUP)

        self.pause_regex = re.compile(r"(?:<p> )?(\(\d*?\.\d*?\))")
        self.pause_value_regex = re.compile(r"\((\d*?\.\d*?)\)")

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

    def sanitize_text_for_tts(self, text, tts_engine):
        text = re.sub(self.pause_regex, "..." if tts_engine == "bark" else "", text)
        text = re.sub(r"(?:\s|\A)i?[hx]+(?=(?:\s|\Z))", "", text, flags=re.IGNORECASE)
        text = re.sub(r"0 ?(?=\[)", "", text)
        text = re.sub("0[.]", "", text)
        if tts_engine == "fastspeech2":
            text = re.sub(r"\[%.*?\]", "", text)
            text = re.sub(r"&=laugh.*?(?=(?:\s|\Z))", "ha! ha! ha!", text, flags=re.IGNORECASE)
            text = re.sub(r"&=.+?(?=(?:\s|\Z))", "", text)
            text = re.sub("yeah[.!?]*", "yeah,", text, flags=re.IGNORECASE)
        elif tts_engine == "bark":
            text = re.sub(r"\[% ", "[", text)
            text = re.sub(r"&=(.+?)(?=(?:\s|\Z))", lambda m: f"[{m.group(1).lower()}]", text)
            text = re.sub(r" +\.\.\.", "...", text)
            text = text.lstrip(".")
        text = re.sub(" {2,}", " ", text)
        text = text.strip()
        return text

    def execute(self, config, device):
        if config is None:
            config = TTSConfig()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.tts_engine == "fastspeech2":
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
            models, cfg, tts_task = load_model_ensemble_and_task_from_hf_hub(
                #"facebook/fastspeech2-en-ljspeech",
                "facebook/fastspeech2-en-200_speaker-cv4",
                arg_overrides={"vocoder": "hifigan", "fp16": False}
            )
            tts_model = models[0].to(device)
            tts_model.encoder.forward = get_encoder_forward_override(tts_model.encoder)
            TTSHubInterface.update_cfg_with_data_cfg(cfg, tts_task.data_cfg)
            tts_generator = tts_task.build_generator(models, cfg)
            tts_generator.vocoder = tts_generator.vocoder.to(device)
            tts_generator.generate = get_generate_override(tts_generator)
        
        elif config.tts_engine == "bark":
            from bark import SAMPLE_RATE
            from bark.generation import _load_history_prompt
            from .bark_api_mod import generate_audio
            from .bark_generation_mod import preload_models
            preload_models(device, text_use_small=True, coarse_use_small=True, fine_use_small=True)
            speaker_history_prompt = dict(_load_history_prompt(self.BARK_PROMPT_LOOKUP[config.speaker]))
            last_input_seg = ""
            last_gen = None
        
        speech_enhancer = SpeechEnhancer(device=device)
        cached_resample = None
        input_buffer = []

        self.running.value = True
        while True:
            try:
                new_config = queue_helpers.skip_queue(self.config_queue)
                if new_config is not None:
                    if new_config.tts_engine == "bark" and new_config.speaker != config.speaker:
                        speaker_history_prompt = dict(_load_history_prompt(self.BARK_PROMPT_LOOKUP[new_config.speaker]))
                        last_input_seg = ""
                        last_gen = None
                    config = new_config

                queue_helpers.transfer_queue_to_buffer(self.input_queue, input_buffer)

                # Determine what items in the input buffer to process:
                # - If the buffer is full, process the whole buffer. 
                # - If the buffer is not full but item(s) exist that contain a pause, 
                #   process everything up to and including the last pause.
                # - Otherwise, do nothing.
                handle_pauses = config.tts_engine != "bark"
                items_to_process = None
                if len(input_buffer) >= config.buffer_size:
                    items_to_process = input_buffer
                elif handle_pauses:
                    idx_after_last_pause = None
                    for i in range(len(input_buffer)-1, -1, -1):
                        if re.search(self.pause_regex, input_buffer[i]):
                            idx_after_last_pause = i+1
                            break
                    if idx_after_last_pause is not None:
                        items_to_process = input_buffer[:idx_after_last_pause]
                        input_buffer = input_buffer[idx_after_last_pause:]

                if items_to_process is not None:
                    next_input = " ".join(items_to_process)
                    items_to_process.clear()
                    # split the input by pauses, converting each pause to blank audio and each non-pause to speech
                    next_input_segs = re.split(self.pause_regex, next_input) if handle_pauses else [next_input]
                    for input_seg in next_input_segs:
                        if handle_pauses and re.match(self.pause_regex, input_seg):
                            # In case of a pause, convert the pause to blank audio
                            pause_value = float(re.search(self.pause_value_regex, input_seg)[1])
                            if not pause_value > 0.0:
                                continue
                            print(f"TTS: Rendering pause: {input_seg}")
                            rate = 22050 // config.downsampling_factor
                            blank_wav = torch.zeros(int(rate * pause_value), dtype=torch.float32).numpy()
                            self.output_queue.put((rate, blank_wav))
                        else:
                            # In case of non-pause text, convert the text to speech
                            input_seg = self.sanitize_text_for_tts(input_seg, config.tts_engine)
                            if not input_seg or not re.sub("[. ]", "", input_seg):
                                print(f"TTS: Skipping: {input_seg}")
                                continue
                            print(f"TTS: Rendering: {input_seg}")
                            if config.tts_engine == "fastspeech2":
                                speaker_id = int(config.speaker.split()[-1])-1
                                sample = TTSHubInterface.get_model_input(tts_task, input_seg, speaker=speaker_id)
                                sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to(device)
                                sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].to(device)
                                if sample["speaker"] is not None:
                                    sample["speaker"] = sample["speaker"].to(device)
                                
                                d_factor = torch.ones_like(sample["net_input"]["src_tokens"], dtype=torch.float32)
                                #d_factor[:, 1] *= config.duration_factor
                                p_factor = torch.ones_like(d_factor)
                                #p_factor[:, 1] *= config.pitch_factor
                                wav, rate = TTSHubInterface.get_prediction(
                                    tts_task, tts_model, tts_generator, sample, d_factor=d_factor, 
                                    p_factor=p_factor, e_factor=config.energy_factor
                                )
                            elif config.tts_engine == "bark":
                                gen_input = " ".join([last_input_seg, input_seg]).lstrip()
                                last_gen, wav = generate_audio(
                                    gen_input, history_prompt=speaker_history_prompt, prefix_prompt=last_gen,
                                    text_temp=0.7, waveform_temp=0.7, silent=True, output_full=True, min_eos_p=0.05, 
                                    max_gen_duration_s=max(1.0, len(input_seg.split()) / 1.25)
                                )
                                wav = torch.tensor((wav * 32767), dtype=torch.float32)
                                rate = SAMPLE_RATE
                                last_input_seg = input_seg

                            # downsample & enhance (if selected)
                            new_rate = rate // config.downsampling_factor
                            wav, cached_resample = audio_helpers.downsample(wav, rate, new_rate, cached_resample)
                            wav, new_rate = speech_enhancer.enhance(config.enhancement_model, wav, new_rate)

                            wav = wav.cpu().numpy()
                            self.output_queue.put((new_rate, wav))
            except Exception as ex:
                #TODO: logging here
                print(ex)
            sleep(0.05)

    def queue_config(self, config):
        self.config_queue.put(config)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        return queue_helpers.join_queue_audio(self.output_queue)
        