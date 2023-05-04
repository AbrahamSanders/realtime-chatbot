import gradio as gr
import torch
import argparse
from time import sleep
from datetime import datetime

from realtime_chatbot.utils import audio_helpers
from realtime_chatbot.tts_handler import TTSHandlerMultiprocessing, TTSConfig
from realtime_chatbot.speech_enhancer import SpeechEnhancer

tts_engine = None
tts_handler = None
speech_enhancer = None

def _synthesize_handler(text, buffer_size):
    text_parts = text.split("|")
    audio_parts = []
    for i, part in enumerate(text_parts):
        tts_handler.queue_input(part)
        if (i+1) % buffer_size == 0 or i == len(text_parts)-1:
            if i == len(text_parts)-1:
                while (i+1) % buffer_size > 0:
                    tts_handler.queue_input("")
                    i += 1
            start = datetime.now()
            output = None
            while output is None:
                output = tts_handler.next_output()
                if (datetime.now()-start).total_seconds() > 60:
                    break
                sleep(0.001)
            if output is not None:
                audio_parts.append(output)
    return audio_parts
    


def process_text(text, buffer_size, voice, downsample_factor, duration_factor, pitch_factor, energy_factor):
    tts_handler.queue_config(TTSConfig(tts_engine=tts_engine, buffer_size=buffer_size, speaker=voice, 
                                       duration_factor=duration_factor, pitch_factor=pitch_factor, energy_factor=energy_factor))
    audio_parts = _synthesize_handler(text, buffer_size)
    if not audio_parts:
        return None, None, None, None, None

    wav_tensor, sr = audio_helpers.concat_audios_to_tensor(audio_parts)

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
    parser = argparse.ArgumentParser("TTS Test")
    parser.add_argument("--tts-engine", type=str, default="fastspeech2", help="TTS engine to use")
    args = parser.parse_args()

    print("\nRunning with arguments:")
    print(args)
    print()

    tts_engine = args.tts_engine
    
    device = torch.device("cuda")
    tts_handler = TTSHandlerMultiprocessing(
        device=device,
        config=TTSConfig(tts_engine=tts_engine),
        wait_until_running=True
    )

    speech_enhancer = SpeechEnhancer(device=device)

    interface = gr.Interface(
        fn=process_text,
        inputs=[
            "text",
            gr.Slider(1, 5, value=1, step=1),
            gr.Dropdown(
                choices=tts_handler.available_speakers, 
                value=tts_handler.available_speakers[0],
                label="Voice"
            ),
            gr.Slider(1, 6, value=1, step=1),
            gr.Slider(-5, 5, value=1, step=0.1),
            gr.Slider(-5, 5, value=1, step=0.1),
            gr.Slider(-5, 5, value=1, step=0.1)
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