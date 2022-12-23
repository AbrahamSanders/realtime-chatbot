import gradio as gr
import torch
from time import sleep
from datetime import datetime

from realtime_chatbot.utils import audio_helpers
from realtime_chatbot.tts_handler import TTSHandlerMultiprocessing, TTSConfig
from realtime_chatbot.speech_enhancer import SpeechEnhancer

tts_handler = None
speech_enhancer = None

def _synthesize_handler(text):
    tts_handler.queue_input(text)
    start = datetime.now()
    while True:
        output = tts_handler.next_output()
        if output:
            return output
        if (datetime.now()-start).total_seconds() > 10:
            return None
        sleep(0.001)

def process_text(text, voice, downsample_factor):
    tts_handler.queue_config(TTSConfig(buffer_size=1, speaker=voice))
    audio = _synthesize_handler(text)

    wav_tensor, sr = audio_helpers.concat_audios_to_tensor([audio])

    sr_downsample = sr // downsample_factor
    wav_downsample, _ = audio_helpers.downsample(wav_tensor, sr, sr_downsample)
    wav_downsample = wav_downsample.numpy()
    
    wav_sources, sr_sources = speech_enhancer.enhance("sepformer-whamr-enhancement", wav_tensor, sr)
    wav_sources = wav_sources.cpu().numpy()

    wav_enhanced_mtl, sr_enhanced_mtl = speech_enhancer.enhance("mtl-mimic-voicebank", wav_tensor, sr)
    wav_enhanced_mtl = wav_enhanced_mtl.cpu().numpy()

    wav_enhanced_metricgan, sr_enhanced_metricgan = speech_enhancer.enhance("metricgan-plus-voicebank", wav_tensor, sr)
    wav_enhanced_metricgan = wav_enhanced_metricgan.cpu().numpy()

    return (sr, wav_tensor.cpu().numpy()), \
           (sr_downsample, wav_downsample), \
           (sr_sources, wav_sources), \
           (sr_enhanced_mtl, wav_enhanced_mtl), \
           (sr_enhanced_metricgan, wav_enhanced_metricgan)

if __name__ == "__main__":
    device = torch.device("cuda")
    tts_handler = TTSHandlerMultiprocessing(
        device=device,
        wait_until_running=True
    )

    speech_enhancer = SpeechEnhancer(device=device)

    interface = gr.Interface(
        fn=process_text,
        inputs=[
            "text",
            gr.Dropdown(
                type="index",
                choices=[f"Voice {i+1}" for i in range(200)],
                value="Voice 16", label="Voice"
            ),
            gr.Slider(1, 6, value=1, step=1)
        ], 
        outputs=[
            gr.Audio(label="Control"),
            gr.Audio(label="Control (downsampled)"),
            gr.Audio(label="Experimental (sepformer WHAMR! separation)"),
            gr.Audio(label="Experimental (Spectral Feature Mapping with mimic)"),
            gr.Audio(label="Experimental (MetricGAN+)")
        ],
        allow_flagging='never',
        title="TTS Test",
        description="TTS Test"
    )
    interface.launch()