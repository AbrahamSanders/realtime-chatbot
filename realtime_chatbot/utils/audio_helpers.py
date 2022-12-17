from audioop import ratecv
from torchaudio.transforms import Resample
import torchaudio
import torch
import numpy as np

def convert_sample_rate(data, original_sr, new_sr): 
    fragments = data.tobytes()
    fragments_new, _ = ratecv(fragments, 2, 1, original_sr, new_sr, None)
    return np.frombuffer(fragments_new, np.int16).flatten().astype(np.float32) / 32768.0

def downsample(data, rate, new_rate, cached_resample=None):
    if new_rate < rate:
        if cached_resample is None or cached_resample.orig_freq != rate or cached_resample.new_freq != new_rate:
            cached_resample = Resample(orig_freq=rate, new_freq=new_rate).to(data.device)
        data = cached_resample(data)
    return data, cached_resample

def concat_audios_to_tensor(buffer, dtype=torch.float32):
    # load from disk or convert to torch.tensor if necessary.
    if isinstance(buffer[0], str):
        buffer = [torchaudio.load(filepath) for filepath in buffer]
    elif isinstance(buffer[0][1], np.ndarray):
        buffer = [(torch.from_numpy(a), sr) for sr, a in buffer]

    # concat the buffer
    sr = buffer[0][1]
    concat_audio = torch.cat([buf[0] for buf in buffer], dim=-1)
    original_dtype = concat_audio.dtype
    concat_audio = concat_audio.to(dtype=dtype)

    # if audio came in as int16, it needs to be scaled after conversion to float32
    if original_dtype == torch.int16 and dtype == torch.float32:
        concat_audio /= 32768.0

    if len(concat_audio.shape) > 1 and concat_audio.shape[0] == 1:
        concat_audio = concat_audio.squeeze(dim=0)
    return concat_audio, sr