import gradio as gr
import torch
import numpy as np
import re
from realtime_chatbot.asr_handler import ASRHandlerMultiprocessing, ASRConfig
from datetime import datetime

class ASRTestGradioInteface:
    def __init__(self):
        self.asr_handler = ASRHandlerMultiprocessing(
            device=torch.device("cuda:1"),
            wait_until_running=False,
            output_debug_audio=True
        )
        self.asr_handler.wait_until_running()

    def update_transcription(self, transcription, new_text, partial_pos):
        if new_text:
            # First, clear out the previous partial segment (if exists)
            if partial_pos > -1:
                transcription = transcription[:partial_pos]
                partial_pos = -1
            # Next, add the new segments to the transcription, 
            # discarding intermediate partial segments.
            new_segments = re.split(" (?=[~*])", new_text)
            for i, seg in enumerate(new_segments):
                if len(seg) > 1 and (seg.startswith("*") or i == len(new_segments)-1):
                    if seg.startswith("~"):
                        partial_pos = len(transcription)
                    if len(transcription) > 0:
                        transcription += " "
                    transcription += seg[1:]
        return transcription, partial_pos

    def execute(self, state, audio, reset, collect, simulate_load, asr_max_buffer_size, asr_model_size, 
                asr_logprob_threshold, asr_no_speech_threshold, asr_lang):
        # queue up configs in case any changes were made.
        asr_config = ASRConfig(model_size=asr_model_size, lang=asr_lang, logprob_threshold=asr_logprob_threshold, 
                               no_speech_threshold=asr_no_speech_threshold, max_buffer_size=asr_max_buffer_size)
        if asr_config != state["asr_config"]:
            state["asr_config"] = asr_config
            self.asr_handler.queue_config(asr_config)

        # If there is collected audio and collect is switched off, output it
        collected_audio = state["collected_audio"]
        collected_audio_concat = None
        if not collect and len(collected_audio) > 0:
            collected_audio_concat = (collected_audio[0][0], np.concatenate([ca[1] for ca in collected_audio]))
            collected_audio.clear()

        # If there is audio input, queue it up for ASR.
        if audio is not None:
            self.asr_handler.queue_input(audio)

        # If there is ASR debug audio output, collect it
        asr_debug_audio = self.asr_handler.next_debug_audio()
        if collect and asr_debug_audio is not None:
            collected_audio.append(asr_debug_audio)

        # If there is ASR output, append to display
        transcription = state["transcription"]
        partial_pos = state["partial_pos"]
        if reset:
            transcription = ""
            partial_pos = -1

        new_text = self.asr_handler.next_output()
        transcription, partial_pos = self.update_transcription(transcription, new_text, partial_pos)

        state["transcription"] = transcription
        state["partial_pos"] = partial_pos

        if simulate_load:
            then = datetime.now()
            while((datetime.now() - then).total_seconds() < 1):
                pass

        return state, transcription, collected_audio_concat

    def launch(self):
        asr_model_size = gr.Dropdown(label="ASR Model size", choices=self.asr_handler.available_model_sizes, value='small.en')
        asr_max_buffer_size_slider = gr.inputs.Slider(minimum=1, maximum=10, default=5, step=1, label="ASR max buffer size")
        asr_logprob_threshold_slider = gr.inputs.Slider(minimum=-3.0, maximum=0.0, default=-0.7, step=0.05, label="ASR Log prob threshold")
        asr_no_speech_threshold_slider = gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.6, step=0.05, label="ASR No speech threshold")

        asr_lang_dropdown = gr.inputs.Dropdown(choices=self.asr_handler.available_languages, label="ASR Language", 
                                               default="English", type="value")

        if asr_lang_dropdown==self.asr_handler.AUTO_DETECT_LANG:
            asr_lang_dropdown=None

        reset_button = gr.Checkbox(value=True, label="Reset")
        collect_button = gr.Checkbox(value=True, label="Collect Audio")
        simulate_load_button = gr.Checkbox(value=False, label="Simulate Load")
        state = gr.State({
            "transcription": "",
            "partial_pos": -1,
            "asr_config": ASRConfig(),
            "collected_audio": []
        })

        output_textbox = gr.Textbox(label="Output")

        interface = gr.Interface(
            fn=self.execute,
            inputs=[
                state,
                gr.Audio(source="microphone", streaming=True, label="ASR Input"),
                reset_button,
                collect_button,
                simulate_load_button,
                asr_max_buffer_size_slider,
                asr_model_size,
                asr_logprob_threshold_slider,
                asr_no_speech_threshold_slider,
                asr_lang_dropdown
                ], 
            outputs=[
                state,
                output_textbox,
                gr.Audio(label="Collected Audio")
            ],
            live=True,
            allow_flagging='never',
            title="ASR Test",
            description="ASR Test"
        )
        interface.launch()


if __name__ == "__main__":
    interface = ASRTestGradioInteface()
    interface.launch()