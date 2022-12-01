import gradio as gr

from realtime_chatbot.realtime_agent import RealtimeAgentMultiprocessing
from realtime_chatbot.tts_handler import TTSHandlerMultiprocessing, TTSConfig
from realtime_chatbot.asr_handler import ASRHandlerMultiprocessing, ASRConfig
from realtime_chatbot.utils import gradio_helpers

class RealtimeAgentGradioInterface:
    def __init__(self):
        self.tts_handler = TTSHandlerMultiprocessing(wait_until_running=False)
        self.agent = RealtimeAgentMultiprocessing(wait_until_running=False, chain_to_input_queue=self.tts_handler.input_queue)
        self.asr_handler = ASRHandlerMultiprocessing(wait_until_running=False, chain_to_input_queue=self.agent.input_queue)
        self.asr_handler.wait_until_running()
        self.agent.wait_until_running()
        self.tts_handler.wait_until_running()
        
        self.audio_html = gradio_helpers.get_audio_html("output_audio")

    def execute(self, audio, state, tts_buffer_size, asr_buffer_size, model_size, logprob_threshold, no_speech_threshold, lang):    
        # queue up ASR & TTS configs in case any changes were made.
        # TODO: only do this if actual changes were made
        asr_config = ASRConfig(model_size=model_size, lang=lang, logprob_threshold=logprob_threshold, 
                               no_speech_threshold=no_speech_threshold, buffer_size=asr_buffer_size)
        tts_config = TTSConfig(buffer_size=tts_buffer_size)
        self.asr_handler.queue_config(asr_config)
        self.tts_handler.queue_config(tts_config)

        # If there is audio input, queue it up for ASR.
        if audio is not None:
            self.asr_handler.queue_input(audio)

        # If there is ASR output, queue it up for the agent and add it to the text chat output.
        dialogue = state["dialogue"]
        transcription_text = self.asr_handler.next_output()
        if transcription_text:
            if "reset" in transcription_text.lower():
                self.agent.reset()
                state["user_speaking"] = None
                dialogue.clear()
            else:
                if state["user_speaking"] is None or not state["user_speaking"]:
                    state["user_speaking"] = True
                    dialogue.append([f"{self.agent.user_identity}:", ""])
                dialogue[-1][0] += f" {transcription_text}"

        # If there is agent output, queue it up for TTS and add it to the text chat output.
        next_output = self.agent.next_output()
        if next_output:
            if state["user_speaking"] is None or state["user_speaking"]:
                state["user_speaking"] = False
                if len(dialogue) == 0:
                    dialogue.append(["", ""])
                dialogue[-1][1] += f"{self.agent.agent_identity}:"
            dialogue[-1][1] += next_output
            
        # If there is TTS output, return the audio along with the HTML hack to make gradio
        # queue it up for autoplay
        next_output_audio = self.tts_handler.next_output()

        return dialogue, next_output_audio, self.audio_html, state
    
    def launch(self):
        title = "Real-time Dialogue Agent"
        description = "Just click Record and start talking to the agent! ASR powered by OpenAI Whisper."

        model_size = gr.Dropdown(label="Model size", choices=self.asr_handler.available_model_sizes, value='medium.en')

        tts_buffer_size_slider = gr.inputs.Slider(minimum=1, maximum=5, default=2, step=1, label="TTS buffer size")
        asr_buffer_size_slider = gr.inputs.Slider(minimum=1, maximum=5, default=3, step=1, label="ASR buffer size")
        logprob_threshold_slider = gr.inputs.Slider(minimum=-3.0, maximum=0.0, default=-0.4, label="Log prob threshold")
        no_speech_threshold_slider = gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.3, label="No speech threshold")

        lang_dropdown = gr.inputs.Dropdown(choices=self.asr_handler.available_languages, label="Language", 
                                           default="English", type="value")

        if lang_dropdown==self.asr_handler.AUTO_DETECT_LANG:
            lang_dropdown=None

        dialogue_chatbot = gr.Chatbot(label="Dialogue").style(color_map=("green", "pink"))

        state = gr.State({"dialogue": [], "user_speaking": None})

        dialogue_interface = gr.Interface(
            fn=self.execute,
            inputs=[
                gr.Audio(source="microphone", streaming=True),
                state,
                tts_buffer_size_slider,
                asr_buffer_size_slider,
                model_size,
                logprob_threshold_slider,
                no_speech_threshold_slider,
                lang_dropdown
                ], 
            outputs=[
                dialogue_chatbot,
                gr.Audio(elem_id="output_audio"),
                gr.HTML(),
                state
            ],
            live=True,
            allow_flagging='never',
            title=title,
            description=description
        )
        dialogue_interface.launch()

if __name__ == "__main__":
    interface = RealtimeAgentGradioInterface()
    interface.launch()