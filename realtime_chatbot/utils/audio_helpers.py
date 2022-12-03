from audioop import ratecv
from torchaudio.transforms import Resample
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