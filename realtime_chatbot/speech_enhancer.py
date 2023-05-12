import numpy as np
import torch
import io
from scipy.io.wavfile import write

from .utils import audio_helpers

class SpeechEnhancer:
    @staticmethod
    def supported_models():
        return [
            "none",
            "denoiser",
            "noisereduce-nonstationary",
            "sepformer-whamr-enhancement", 
            "mtl-mimic-voicebank", 
            "metricgan-plus-voicebank",
            "damo/speech_frcrn_ans_cirm_16k"
        ]

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model_dict = {}
        self.cached_resample_dict = {}
    
    def _get_model(self, model_name):
        if model_name in self.model_dict:
            return self.model_dict[model_name]
        
        if model_name == "denoiser":
            from denoiser import pretrained
            model = pretrained.dns64().to(self.device)
        elif model_name == "noisereduce-nonstationary":
            from noisereduce.torchgate import TorchGate
            model = TorchGate(sr=16000, nonstationary=True).to(self.device)
        elif model_name == "sepformer-whamr-enhancement":
            from speechbrain.pretrained import SepformerSeparation
            model = SepformerSeparation.from_hparams(
                source=f"speechbrain/{model_name}", 
                savedir=f"speechbrain/pretrained_models/{model_name}",
                run_opts={"device":self.device}
            )
        elif model_name == "mtl-mimic-voicebank":
            from speechbrain.pretrained import WaveformEnhancement
            model = WaveformEnhancement.from_hparams(
                source=f"speechbrain/{model_name}",
                savedir=f"speechbrain/pretrained_models/{model_name}",
                run_opts={"device":self.device}
            )
        elif model_name == "metricgan-plus-voicebank":
            from speechbrain.pretrained import SpectralMaskEnhancement
            model = SpectralMaskEnhancement.from_hparams(
                source=f"speechbrain/{model_name}",
                savedir=f"speechbrain/pretrained_models/{model_name}",
                run_opts={"device":self.device}
            )
        elif model_name == "damo/speech_frcrn_ans_cirm_16k":
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            model = pipeline(
                Tasks.acoustic_noise_suppression,
                model='damo/speech_frcrn_ans_cirm_16k',
                device=f"{self.device.type}:{self.device.index}" if self.device.index is not None \
                    else self.device.type
            )
        else:
            model = None
        
        self.model_dict[model_name] = model
        return model

    def enhance(self, model_name, audio_tensor, sr):
        model = self._get_model(model_name)
        if model is None:
            return audio_tensor, sr

        audio_tensor = audio_tensor.unsqueeze(dim=0).to(self.device)
        
        if model_name == "denoiser":
            new_sr = model.sample_rate
        elif model_name == "noisereduce-nonstationary":
            new_sr = model.sr
        elif model_name == "damo/speech_frcrn_ans_cirm_16k":
            new_sr = model.SAMPLE_RATE
        else:
            new_sr = model.hparams.sample_rate
        
        cached_resample = self.cached_resample_dict.get(model_name, None)
        audio_tensor, cached_resample = audio_helpers.downsample(audio_tensor, sr, new_sr, cached_resample)
        self.cached_resample_dict[model_name] = cached_resample

        if model_name == "denoiser":
            with torch.no_grad():
                audio_tensor = model(audio_tensor)
            audio_tensor = audio_tensor[0, 0]
        elif model_name == "noisereduce-nonstationary":
            with torch.no_grad():
                audio_tensor = model(audio_tensor)
            audio_tensor = audio_tensor[0]
        elif model_name == "sepformer-whamr-enhancement":
            audio_tensor = model.separate_batch(audio_tensor)
            audio_tensor = audio_tensor[0, :, 0]
        elif model_name == "mtl-mimic-voicebank":
            audio_tensor = model.enhance_batch(audio_tensor)
            audio_tensor = audio_tensor[0]
        elif model_name == "metricgan-plus-voicebank":
            audio_tensor = model.enhance_batch(audio_tensor, lengths=torch.tensor([1.]))
            audio_tensor = audio_tensor[0]
        elif model_name == "damo/speech_frcrn_ans_cirm_16k":
            audio_bytes_io = io.BytesIO(bytes())
            write(audio_bytes_io, new_sr, audio_tensor[0].cpu().numpy())
            model_output = model(audio_bytes_io.read())
            wav = np.frombuffer(model_output["output_pcm"], dtype=np.int16)
            audio_tensor = torch.tensor(wav, dtype=audio_tensor.dtype, device=audio_tensor.device) / 32768.0
        else:
            raise ValueError(f"Unsupported model class {type(model).__name__}")

        return audio_tensor, new_sr