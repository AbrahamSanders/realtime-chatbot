from speechbrain.pretrained import SepformerSeparation, WaveformEnhancement, SpectralMaskEnhancement
import torch
from .utils import audio_helpers

class SpeechEnhancer:
    @staticmethod
    def supported_models():
        return [
            "none",
            "sepformer-whamr-enhancement", 
            "mtl-mimic-voicebank", 
            "metricgan-plus-voicebank"
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
        
        if model_name == "sepformer-whamr-enhancement":
            model = SepformerSeparation.from_hparams(
                source=f"speechbrain/{model_name}", 
                savedir=f"speechbrain/pretrained_models/{model_name}",
                run_opts={"device":self.device}
            )
        elif model_name == "mtl-mimic-voicebank":
            model = WaveformEnhancement.from_hparams(
                source=f"speechbrain/{model_name}",
                savedir=f"speechbrain/pretrained_models/{model_name}",
                run_opts={"device":self.device}
            )
        elif model_name == "metricgan-plus-voicebank":
            model = SpectralMaskEnhancement.from_hparams(
                source=f"speechbrain/{model_name}",
                savedir=f"speechbrain/pretrained_models/{model_name}",
                run_opts={"device":self.device}
            )
        else:
            model = None
        
        self.model_dict[model_name] = model
        return model

    def enhance(self, model_name, audio_tensor, sr):
        model = self._get_model(model_name)
        if model is None:
            return audio_tensor, sr

        audio_tensor = audio_tensor.to(self.device)
        new_sr = model.hparams.sample_rate
        cached_resample = self.cached_resample_dict.get(model_name, None)
        audio_tensor, cached_resample = audio_helpers.downsample(audio_tensor, sr, new_sr, cached_resample)
        self.cached_resample_dict[model_name] = cached_resample

        audio_tensor = audio_tensor.unsqueeze(dim=0)
        if isinstance(model, SepformerSeparation):
            audio_tensor = model.separate_batch(audio_tensor)
            audio_tensor = audio_tensor[0, :, 0]
        elif isinstance(model, WaveformEnhancement):
            audio_tensor = model.enhance_batch(audio_tensor)
            audio_tensor = audio_tensor[0]
        elif isinstance(model, SpectralMaskEnhancement):
            audio_tensor = model.enhance_batch(audio_tensor, lengths=torch.tensor([1.]))
            audio_tensor = audio_tensor[0]
        else:
            raise ValueError(f"Unsupported model class {type(model).__name__}")

        return audio_tensor, new_sr