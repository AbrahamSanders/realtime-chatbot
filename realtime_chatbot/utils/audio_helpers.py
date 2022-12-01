from audioop import ratecv
import numpy as np

def convert_sample_rate(data, original_sr, new_sr): 
    fragments = data.tobytes()
    fragments_new, _ = ratecv(fragments, 2, 1, original_sr, new_sr, None)
    return np.frombuffer(fragments_new, np.int16).flatten().astype(np.float32) / 32768.0