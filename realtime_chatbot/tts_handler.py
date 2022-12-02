from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from torchaudio.transforms import Resample
import g2p_en
import torch
import re
from typing import Optional
from time import sleep
from datetime import datetime

from .utils import queue_helpers

class TTSConfig:
    def __init__(self, buffer_size=3, downsampling_factor=2):
        self.buffer_size = buffer_size
        self.downsampling_factor = downsampling_factor

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class TTSHandlerMultiprocessing:
    def __init__(self, wait_until_running=True, config=None):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.config_queue = ctx.SimpleQueue()
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.running = ctx.Value(c_bool, False)

        self.execute_process = ctx.Process(target=self.execute, daemon=True, args=(config,))
        self.execute_process.start()

        if wait_until_running:
            self.wait_until_running()

    def wait_until_running(self):
        #TODO: use an Event instead of a loop
        while not self.is_running():
            sleep(0.01)

    def is_running(self):
        return self.running.value

    def tts_phonemize(
        self,
        text: str,
        lang: Optional[str],
        phonemizer: Optional[str] = None,
        preserve_punct: bool = False,
        to_simplified_zh: bool = False,
    ):
        if preserve_punct:
            return " ".join("|" if p == " " else p for p in self.g2p(text))
        else:
            res = [{",": "sp", ";": "sp"}.get(p, p) for p in self.g2p(text)]
            return " ".join(p for p in res if p.isalnum())

    def sanitize_text_for_tts(self, text):
        text = re.sub(r"\(\d*?\.\d*?\)", "", text)
        text = re.sub("[hx]{2,}", "", text)
        text = re.sub(" {2,}", " ", text)
        text = text.strip()
        return text

    def execute(self, config):
        if config is None:
            config = TTSConfig()
        self.g2p = g2p_en.G2p()
        TTSHubInterface.phonemize = self.tts_phonemize
        tts_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models, cfg, tts_task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False}
        )
        tts_model = models[0].to(tts_device)
        TTSHubInterface.update_cfg_with_data_cfg(cfg, tts_task.data_cfg)
        tts_generator = tts_task.build_generator(models, cfg)
        tts_generator.vocoder = tts_generator.vocoder.to(tts_device)
        downsample = None
        input_buffer = []
        last_output_time = datetime.now()

        self.running.value = True
        while True:
            try:
                new_config = queue_helpers.skip_queue(self.config_queue)
                if new_config is not None:
                    config = new_config

                queue_helpers.transfer_queue_to_buffer(self.input_queue, input_buffer)

                seconds_since_last_output = (datetime.now() - last_output_time).total_seconds()
                buffer_size = config.buffer_size if seconds_since_last_output < config.buffer_size \
                                                 else config.buffer_size // 2 + 1
                if len(input_buffer) >= buffer_size:
                    next_input = " ".join(input_buffer)
                    input_buffer.clear()
                    next_input = self.sanitize_text_for_tts(next_input)
                    if next_input:
                        sample = TTSHubInterface.get_model_input(tts_task, next_input)
                        sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to(tts_device)
                        sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].to(tts_device)
                        
                        wav, rate = TTSHubInterface.get_prediction(tts_task, tts_model, tts_generator, sample)
                        new_rate = rate // config.downsampling_factor
                        if new_rate < rate:
                            if downsample is None or downsample.orig_freq != rate or downsample.new_freq != new_rate:
                                downsample = Resample(orig_freq=rate, new_freq=new_rate).to(tts_device)
                            wav = downsample(wav)
                        wav = wav.cpu().numpy()
                        
                        self.output_queue.put((new_rate, wav))
                        last_output_time = datetime.now()
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
        