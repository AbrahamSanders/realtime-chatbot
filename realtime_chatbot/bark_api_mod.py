from typing import Dict, Optional, Union

import numpy as np

from bark.generation import codec_decode, generate_coarse, generate_fine
from .bark_generation_mod import generate_text_semantic

def text_to_semantic(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    prefix_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
    **kwargs
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        prefix_prompt=prefix_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True,
        **kwargs
    )
    return x_semantic


def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
    **coarse_kwargs
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True,
        **coarse_kwargs
    )
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr

def concat_prompts(history_prompt, prefix_prompt):
    combined_prompt = {}
    for key in history_prompt:
        combined_prompt[key] = np.concatenate((history_prompt[key], prefix_prompt[key]), axis=-1)
    return combined_prompt

def generate_audio(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    prefix_prompt: Optional[Union[Dict, str]] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
    min_eos_p: float = 0.2, 
    max_gen_duration_s: float = 14.0,
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    semantic_tokens = text_to_semantic(
        text,
        history_prompt=history_prompt,
        prefix_prompt=prefix_prompt,
        temp=text_temp,
        silent=silent,
        min_eos_p=min_eos_p,
        max_gen_duration_s=max_gen_duration_s,
    )

    if prefix_prompt is not None:
        history_prompt = concat_prompts(history_prompt, prefix_prompt)

    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out
    return audio_arr