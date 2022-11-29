import numpy as np
import gradio as gr
import whisper
from whisper import tokenizer, _MODELS
from whisper.audio import SAMPLE_RATE
from audioop import ratecv
import time
from collections import deque

from realtime_chatbot.realtime_agent import RealtimeAgentMultiprocessing

class RealtimeAgentGradioInterface:
    def __init__(self):
        self.current_size = 'small.en'
        self.model = whisper.load_model(self.current_size)
        self.agent = RealtimeAgentMultiprocessing()
        self.AUTO_DETECT_LANG = "Auto Detect"

    def convert_sample_rate(self, data, original_sr, new_sr): 
        fragments = data.tobytes()
        fragments_new, _ = ratecv(fragments, 2, 1, original_sr, new_sr, None)
        return np.frombuffer(fragments_new, np.int16).flatten().astype(np.float32) / 32768.0

    def transcribe(self, audio, state, model_size, delay, logprob_threshold, no_speech_threshold, lang):
        if delay > 1.0:
            time.sleep(delay - 1)

        if model_size != self.current_size:
            self.current_size = model_size
            self.model = whisper.load_model(self.current_size)
    
        dialogue = state["dialogue"]
        if audio is not None:
            audio = self.convert_sample_rate(audio[1], audio[0], SAMPLE_RATE)
            last_n_segs = state["last_n_segs"]
            initial_prompt = " ".join([seg for seg in last_n_segs if seg])
            transcription = self.model.transcribe(
                audio,
                language = lang if lang != self.AUTO_DETECT_LANG else None,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                initial_prompt = initial_prompt
            )
            transcription_text = transcription['text'].strip()
            if len(last_n_segs) == last_n_segs.maxlen:
                last_n_segs.popleft()
            last_n_segs.append(transcription_text)

            if transcription_text:
                if "reset" in transcription_text.lower():
                    self.agent.reset()
                    state["user_speaking"] = None
                    dialogue.clear()
                else:
                    self.agent.queue_input(transcription_text)
                    if state["user_speaking"] is None or not state["user_speaking"]:
                        state["user_speaking"] = True
                        dialogue.append([f"{self.agent.user_identity}:", ""])
                    dialogue[-1][0] += f" {transcription_text}"

            next_output = self.agent.next_output()
            if next_output:
                if state["user_speaking"] is None or state["user_speaking"]:
                    state["user_speaking"] = False
                    if len(dialogue) == 0:
                        dialogue.append(["", ""])
                    dialogue[-1][1] += f"{self.agent.agent_identity}:"
                dialogue[-1][1] += next_output

        return dialogue, state
    
    def launch(self):
        title = "Real-time Dialogue Agent"
        description = "Just click Record and start talking to the agent! ASR powered by OpenAI Whisper."

        model_size = gr.Dropdown(label="Model size", choices=list(_MODELS), value='small.en')

        delay_slider = gr.inputs.Slider(minimum=1, maximum=5, default=1.2, label="Rate of transcription")
        logprob_threshold_slider = gr.inputs.Slider(minimum=-3.0, maximum=0.0, default=-0.4, label="Log prob threshold")
        no_speech_threshold_slider = gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.3, label="No speech threshold")

        available_languages = sorted(tokenizer.TO_LANGUAGE_CODE.keys())
        available_languages = [lang.capitalize() for lang in available_languages]
        available_languages = [self.AUTO_DETECT_LANG]+available_languages

        lang_dropdown = gr.inputs.Dropdown(choices=available_languages, label="Language", default="English", type="value")

        if lang_dropdown==self.AUTO_DETECT_LANG:
            lang_dropdown=None

        dialogue_chatbot = gr.Chatbot(label="Dialogue").style(color_map=("green", "pink"))

        state = gr.State({"dialogue": [], "user_speaking": None, "last_n_segs": deque(maxlen=1)})

        gr.Interface(
            fn=self.transcribe,
            inputs=[
                gr.Audio(source="microphone", type="numpy", streaming=True),
                state,
                model_size,
                delay_slider,
                logprob_threshold_slider,
                no_speech_threshold_slider,
                lang_dropdown
                ], 
            outputs=[
                dialogue_chatbot,
                state
            ],
            live=True,
            allow_flagging='never',
            title=title,
            description=description,
        ).launch(
            # enable_queue=True,
            # debug=True
        )

if __name__ == "__main__":
    interface = RealtimeAgentGradioInterface()
    interface.launch()