import gradio as gr
import numpy as np
import torch
from whisper.audio import SAMPLE_RATE
from realtime_chatbot.utils import audio_helpers
from realtime_chatbot.asr_handler import ASRHandlerMultiprocessing, ASRConfig
from datetime import datetime
from time import sleep
import whisper

model = None
asr_handler = None

def _process_audio_torchaudio(audio, cached_resample=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_tensor, sr = audio_helpers.concat_audios_to_tensor([audio])
    audio_tensor = audio_tensor.to(device)
    audio_torchaudio, cached_resample = audio_helpers.downsample(audio_tensor, sr, SAMPLE_RATE, cached_resample)
    audio_torchaudio = (SAMPLE_RATE, audio_torchaudio)
    return audio_torchaudio, cached_resample

def _process_audio_ratecv(audio):
    audio_ratecv = audio_helpers.convert_sample_rate(audio[1], audio[0], SAMPLE_RATE)
    audio_ratecv = (SAMPLE_RATE, audio_ratecv)
    return audio_ratecv

def _transcribe(audio):
    audio = audio[1]
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio).to(model.device)
    transcription = model.transcribe(
        audio,
        language = "en",
        logprob_threshold=-0.4,
        no_speech_threshold=0.3
    )
    transcription_text = transcription['text'].strip()
    return transcription_text

def _transcribe_handler(audio):
    asr_handler.queue_input(audio)
    start = datetime.now()
    while True:
        output = asr_handler.next_output()
        if output:
            return output
        if (datetime.now()-start).total_seconds() > 10:
            return ""
        sleep(0.001)
        

def process_audio(audio_numpy, audio_filepath):
    a, cached_resample = _process_audio_torchaudio(audio_filepath)
    _ = _process_audio_ratecv(audio_numpy)
    _ = _transcribe_handler(audio_filepath)
    _ = _transcribe(a)

    start = datetime.now()
    audio_torchaudio_numpy, _ = _process_audio_torchaudio(audio_numpy, cached_resample)
    torchaudio_numpy_transcript = _transcribe(audio_torchaudio_numpy)
    torchaudio_numpy_time = (datetime.now()-start).total_seconds()

    start = datetime.now()
    audio_torchaudio_filepath, _ = _process_audio_torchaudio(audio_filepath, cached_resample)
    torchaudio_filepath_transcript = _transcribe(audio_torchaudio_filepath)
    torchaudio_filepath_time = (datetime.now()-start).total_seconds()
    
    start = datetime.now()
    audio_ratecv = _process_audio_ratecv(audio_numpy)
    ratecv_transcript = _transcribe(audio_ratecv)
    ratecv_time = (datetime.now()-start).total_seconds()

    start = datetime.now()
    torchaudio_numpy_handler_transcript = _transcribe_handler(audio_numpy)
    torchaudio_numpy_handler_time = (datetime.now()-start).total_seconds()

    start = datetime.now()
    torchaudio_filepath_handler_transcript = _transcribe_handler(audio_filepath)
    torchaudio_filepath_handler_time = (datetime.now()-start).total_seconds()

    audio_torchaudio_numpy = (audio_torchaudio_numpy[0], audio_torchaudio_numpy[1].cpu().numpy())
    audio_torchaudio_filepath = (audio_torchaudio_filepath[0], audio_torchaudio_filepath[1].cpu().numpy())
    info = f"Diff (torchaudio numpy vs filepath): {np.max(np.abs(audio_torchaudio_numpy[1]-audio_torchaudio_filepath[1]))}\n\n" \
           f"Diff (torchaudio numpy vs ratecv): {np.max(np.abs(audio_torchaudio_numpy[1]-audio_ratecv[1]))}\n\n" \
           f"Diff (torchaudio filepath vs ratecv): {np.max(np.abs(audio_torchaudio_filepath[1]-audio_ratecv[1]))}\n\n" \
           f"Time (torchaudio numpy): {torchaudio_numpy_time}\n\n" \
           f"Time (torchaudio filepath): {torchaudio_filepath_time}\n\n" \
           f"Time (ratecv): {ratecv_time}\n\n" \
           f"Time (handler numpy): {torchaudio_numpy_handler_time}\n\n" \
           f"Time (handler filepath): {torchaudio_filepath_handler_time}\n\n" \
           f"Transcript (torchaudio numpy): \"{torchaudio_numpy_transcript}\"\n\n" \
           f"Transcript (torchaudio filepath): \"{torchaudio_filepath_transcript}\"\n\n" \
           f"Transcript (ratecv): \"{ratecv_transcript}\"\n\n" \
           f"Transcript (handler numpy): \"{torchaudio_numpy_handler_transcript}\"\n\n" \
           f"Transcript (handler filepath): \"{torchaudio_filepath_handler_transcript}\""

    return audio_torchaudio_numpy, audio_torchaudio_filepath, audio_ratecv, info

if __name__ == "__main__":
    model = whisper.load_model("small.en")

    asr_handler = ASRHandlerMultiprocessing(
            device=torch.device("cuda:1"),
            wait_until_running=False,
            config=ASRConfig(buffer_size=1)
        )
    asr_handler.wait_until_running()

    interface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(label="Input (numpy)"),
            gr.Audio(label="Input (filepath)", type="filepath"), 
            ], 
        outputs=[
            gr.Audio(label="Output (torchaudio, numpy)"),
            gr.Audio(label="Output (torchaudio, filepath)"),
            gr.Audio(label="Output (ratecv, numpy)"),
            "markdown"
        ],
        allow_flagging='never',
        title="Audio Test",
        description="Audio Test"
    )
    interface.launch()