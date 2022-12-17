import gradio as gr
import torch
import numpy as np
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

    def execute(self, state, audio, reset, collect, simulate_load, asr_buffer_size, asr_model_size, 
                asr_logprob_threshold, asr_no_speech_threshold, asr_lang):
        # queue up configs in case any changes were made.
        asr_config = ASRConfig(model_size=asr_model_size, lang=asr_lang, logprob_threshold=asr_logprob_threshold, 
                               no_speech_threshold=asr_no_speech_threshold, buffer_size=asr_buffer_size)
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
        if reset:
            transcription.clear()

        transcription_text = self.asr_handler.next_output()
        if transcription_text:
            transcription.append(transcription_text)

        if simulate_load:
            then = datetime.now()
            while((datetime.now() - then).total_seconds() < 1):
                pass

        return state, " ".join(transcription), collected_audio_concat, "\n\n".join(transcription)

    def launch(self):
        asr_model_size = gr.Dropdown(label="ASR Model size", choices=self.asr_handler.available_model_sizes, value='small.en')
        asr_buffer_size_slider = gr.inputs.Slider(minimum=1, maximum=5, default=3, step=1, label="ASR buffer size")
        asr_logprob_threshold_slider = gr.inputs.Slider(minimum=-3.0, maximum=0.0, default=-0.4, step=0.05, label="ASR Log prob threshold")
        asr_no_speech_threshold_slider = gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.3, step=0.05, label="ASR No speech threshold")

        asr_lang_dropdown = gr.inputs.Dropdown(choices=self.asr_handler.available_languages, label="ASR Language", 
                                               default="English", type="value")

        if asr_lang_dropdown==self.asr_handler.AUTO_DETECT_LANG:
            asr_lang_dropdown=None

        reset_button = gr.Checkbox(value=True, label="Reset")
        collect_button = gr.Checkbox(value=True, label="Collect Audio")
        simulate_load_button = gr.Checkbox(value=False, label="Simulate Load")
        state = gr.State({
            "transcription": [], 
            "asr_config": ASRConfig(),
            "collected_audio": []
        })

        output_textbox = gr.Textbox(label="Output")
        output_markdown = gr.Markdown(label="Segments")

        interface = gr.Interface(
            fn=self.execute,
            inputs=[
                state,
                gr.Audio(source="microphone", streaming=True, label="ASR Input"),
                reset_button,
                collect_button,
                simulate_load_button,
                asr_buffer_size_slider,
                asr_model_size,
                asr_logprob_threshold_slider,
                asr_no_speech_threshold_slider,
                asr_lang_dropdown
                ], 
            outputs=[
                state,
                output_textbox,
                gr.Audio(label="Collected Audio"),
                output_markdown
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