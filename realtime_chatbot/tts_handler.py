from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import g2p_en
import torch
import re
from time import sleep

from .utils import queue_helpers, audio_helpers
from .speech_enhancer import SpeechEnhancer
from .tts_overrides import (
    get_phonemize_override, get_get_prediction_override, 
    get_generate_override, get_encoder_forward_override
)

class TTSConfig:
    def __init__(self, buffer_size=4, downsampling_factor=1, speaker=0, enhancement_model="none",
                 duration_factor=1.0, pitch_factor=1.0, energy_factor=1.0):
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

        self.pause_regex = re.compile(r"(?:<p> )?\(\d*?\.\d*?\)")

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

    def sanitize_text_for_tts(self, text):
        text = re.sub(self.pause_regex, "", text)
        text = re.sub(r"(?:\s|\A)i?[hx]+(?=(?:\s|\Z))", "", text, flags=re.IGNORECASE)
        text = re.sub(r"0 ?(?=\[)", "", text)
        text = re.sub("0[.]", "", text)
        text = re.sub(r"\[%.*?\]", "", text)
        text = re.sub(r"&=laugh.*?(?=(?:\s|\Z))", "ha! ha! ha!", text, flags=re.IGNORECASE)
        text = re.sub(r"&=.+?(?=(?:\s|\Z))", "", text)
        text = re.sub("yeah[.!?]*", "yeah,", text, flags=re.IGNORECASE)
        text = re.sub(" {2,}", " ", text)
        text = text.strip()
        return text

    def execute(self, config, device):
        if config is None:
            config = TTSConfig()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g2p = g2p_en.G2p()
        TTSHubInterface.phonemize = get_phonemize_override(TTSHubInterface, g2p)
        TTSHubInterface.get_prediction = get_get_prediction_override(TTSHubInterface)
        models, cfg, tts_task = load_model_ensemble_and_task_from_hf_hub(
            #"facebook/fastspeech2-en-ljspeech",
            "facebook/fastspeech2-en-200_speaker-cv4",
            arg_overrides={"vocoder": "hifigan", "fp16": True}
        )
        tts_model = models[0].to(device)
        tts_model.encoder.forward = get_encoder_forward_override(tts_model.encoder)
        TTSHubInterface.update_cfg_with_data_cfg(cfg, tts_task.data_cfg)
        tts_generator = tts_task.build_generator(models, cfg)
        tts_generator.vocoder = tts_generator.vocoder.to(device)
        tts_generator.generate = get_generate_override(tts_generator)
        speech_enhancer = SpeechEnhancer(device=device)
        cached_resample = None
        input_buffer = []

        self.running.value = True
        while True:
            try:
                new_config = queue_helpers.skip_queue(self.config_queue)
                if new_config is not None:
                    config = new_config

                queue_helpers.transfer_queue_to_buffer(self.input_queue, input_buffer)

                # Determine what items in the input buffer to process:
                # - If the buffer is full, process the whole buffer. 
                # - If the buffer is not full but item(s) exist that contain a pause, 
                #   process everything up to and including the last pause.
                # - Otherwise, do nothing.
                items_to_process = None
                if len(input_buffer) >= config.buffer_size:
                    items_to_process = input_buffer
                else:
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
                    next_input = self.sanitize_text_for_tts(next_input)
                    if next_input:
                        sample = TTSHubInterface.get_model_input(tts_task, next_input, speaker=config.speaker)
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

                        # downsample & enhance (if selected)
                        new_rate = rate // config.downsampling_factor
                        wav, cached_resample = audio_helpers.downsample(wav, rate, new_rate, cached_resample)
                        wav, new_rate = speech_enhancer.enhance(config.enhancement_model, wav, new_rate)

                        wav = wav.cpu().numpy()
                        self.output_queue.put((new_rate, wav))
            except:
                #TODO: logging here
                pass
            sleep(0.05)

    def queue_config(self, config):
        self.config_queue.put(config)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        return queue_helpers.join_queue_audio(self.output_queue)
        