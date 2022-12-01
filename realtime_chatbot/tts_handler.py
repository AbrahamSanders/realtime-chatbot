from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import g2p_en
import torch
import re
from typing import Optional
from time import sleep

from .utils import queue_helpers

class TTSHandlerMultiprocessing:
    def __init__(self, wait_until_running=True):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.input_queue = ctx.SimpleQueue()
        self.output_queue = ctx.SimpleQueue()
        self.running = ctx.Value(c_bool, False)

        self.execute_process = ctx.Process(target=self.execute, daemon=True)
        self.execute_process.start()

        if wait_until_running:
            #TODO: use an Event instead of a loop
            while not self.is_running():
                sleep(0.01)

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

    def execute(self):
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

        self.running.value = True
        while True:
            try:
                next_input = queue_helpers.join_queue(self.input_queue)
                if next_input:
                    next_input = self.sanitize_text_for_tts(next_input)
                if next_input:
                    sample = TTSHubInterface.get_model_input(tts_task, next_input)
                    sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to(tts_device)
                    sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].to(tts_device)
                    wav, rate = TTSHubInterface.get_prediction(tts_task, tts_model, tts_generator, sample)
                    wav = wav.cpu().numpy()
                    self.output_queue.put((rate, wav))
            except:
                #TODO: logging here
                pass
            sleep(0.01)

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self):
        if not self.output_queue.empty():
            return self.output_queue.get()
        return None

    def is_running(self):
        return self.running.value

