import gradio as gr
import re
import uuid

from realtime_chatbot.realtime_agent import RealtimeAgentMultiprocessing, RealtimeAgentConfig
from realtime_chatbot.tts_handler import TTSHandlerMultiprocessing, TTSConfig
from realtime_chatbot.asr_handler import ASRHandlerMultiprocessing, ASRConfig
from realtime_chatbot.utils import gradio_helpers, device_helpers, args_helpers
from realtime_chatbot.identity import Identity
from realtime_chatbot.speech_enhancer import SpeechEnhancer

class RealtimeAgentGradioInterface:
    def __init__(self, args):
        self.args = args
        device_map = device_helpers.get_device_map()
        self.tts_handler = TTSHandlerMultiprocessing(
            wait_until_running=False,
            config=TTSConfig(tts_engine=self.args.tts_engine),
            device=device_map["tts"]
        )
        self.agent = RealtimeAgentMultiprocessing(
            wait_until_running=False,
            config=RealtimeAgentConfig(random_state=self.args.random_state, 
                                       prevent_special_token_generation=self.args.prevent_special_token_generation,
                                       add_special_pause_token=self.args.add_special_pause_token,
                                       debug=self.args.debug),
            modelpath=self.args.agent_modelpath,
            device=device_map["agent"],
            chain_to_input_queue=self.tts_handler.input_queue, 
            output_sequence=True,
            output_sequence_max_length=5000
        )
        self.asr_handler = ASRHandlerMultiprocessing(
            wait_until_running=False,
            device=device_map["asr"],
            chain_to_input_queue=self.agent.input_queue
        )
        self.asr_handler.wait_until_running()
        self.agent.wait_until_running()
        self.tts_handler.wait_until_running()
        
        self.audio_html = gradio_helpers.get_audio_html("reset_button", "output_audio")
        
        self.any_identity_regex = re.compile(r"S\d+?:")
        self.sequence_split_regex = re.compile(rf"\s(?={self.any_identity_regex.pattern})")

    def dialogue_from_sequence(self, sequence, agent_config):
        dialogue = re.split(self.sequence_split_regex, sequence)[1:]
        dialogue_unflattened = []
        for utt in dialogue:
            if utt.startswith(agent_config.user_identity):
                dialogue_unflattened.append([utt, ""])
            else:
                if len(dialogue_unflattened) == 0:
                    dialogue_unflattened.append(["", ""])
                if len(dialogue_unflattened[-1][1]) > 0:
                    utt = re.sub(self.any_identity_regex, "", utt)
                dialogue_unflattened[-1][1] += utt
        return dialogue_unflattened

    def execute(self, state, audio, summary, opening_utt, reset, agent_starts, agent_interval, similarity_threshold, 
                min_penalty_alpha, max_penalty_alpha, sample_top_p,
                tts_downsampling_factor, tts_buffer_size, tts_enhancement,
                asr_max_buffer_size, asr_model_size, asr_logprob_threshold, asr_no_speech_threshold, asr_lang,
                user_name, user_age, user_sex, agent_name, agent_age, agent_sex, agent_voice):

        # queue up configs in case any changes were made.
        asr_config = ASRConfig(model_size=asr_model_size, lang=asr_lang, logprob_threshold=asr_logprob_threshold, 
                               no_speech_threshold=asr_no_speech_threshold, max_buffer_size=asr_max_buffer_size)
        if asr_config != state["asr_config"]:
            state["asr_config"] = asr_config
            self.asr_handler.queue_config(asr_config)

        agent_config = RealtimeAgentConfig(
            interval=agent_interval, identities={
                "S1": Identity(user_name, user_age, user_sex),
                "S2": Identity(agent_name, agent_age, agent_sex)
            }, 
            random_state=self.args.random_state, 
            summary=summary,
            prevent_special_token_generation=self.args.prevent_special_token_generation,
            add_special_pause_token=self.args.add_special_pause_token,
            predictive_lookahead=similarity_threshold < 1.0,
            similarity_threshold=similarity_threshold,
            min_penalty_alpha=min_penalty_alpha,
            max_penalty_alpha=max_penalty_alpha,
            sample_top_p=sample_top_p,
            agent_starts_transcript=agent_starts,
            opening_utterance=opening_utt,
            debug=self.args.debug
        )
        if agent_config != state["agent_config"]:
            state["agent_config"] = agent_config
            self.agent.queue_config(agent_config)

        tts_config = TTSConfig(tts_engine=self.args.tts_engine, buffer_size=tts_buffer_size, 
                               downsampling_factor=tts_downsampling_factor, 
                               speaker=agent_voice, enhancement_model=tts_enhancement)
        if tts_config != state["tts_config"]:
            state["tts_config"] = tts_config
            self.tts_handler.queue_config(tts_config)

        # If there is audio input, queue it up for ASR.
        if audio is not None:
            self.asr_handler.queue_input(audio)
            #print("Queued Audio for ASR")

        # If there is ASR output, check for reset event
        transcription_text = self.asr_handler.next_output()
        if transcription_text:
            if "reset" in transcription_text.lower():
                reset = True

        # If there is agent output, add it to the text chat output.
        _ = self.agent.next_output()
        next_sequence = self.agent.next_sequence()
        if next_sequence:
            state["dialogue"] = self.dialogue_from_sequence(next_sequence, agent_config)
            
        # If there is TTS output, return the audio along with the HTML hack to make gradio
        # queue it up for autoplay
        next_output_audio = self.tts_handler.next_output()

        # If reset flag is set, queue an agent reset
        if reset:
            self.agent.queue_reset()

        #print (f"Gradio loop done: {str(uuid.uuid4())[:8]}")
        return state, state["dialogue"], next_output_audio, self.audio_html
    
    def launch(self):
        title = "Real-time Dialogue Agent"
        description = f"Just click 'Record', uncheck 'Reset', and start talking! ---- (Agent: {self.args.agent_modelpath}; " \
                      f"ASR: whisper; TTS: {self.args.tts_engine})"

        asr_model_size = gr.Dropdown(label="ASR Model size", choices=self.asr_handler.available_model_sizes, value='base.en')

        agent_interval_slider = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, label="Agent prediction interval",
                                          value=1.3 if self.args.tts_engine == "bark" else 0.8)
        similarity_threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.01, label="Predictive lookahead similarity threshold (1.0 to disable lookahead)")

        
        min_penalty_alpha_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Min Penalty-alpha")
        max_penalty_alpha_slider = gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="Max Penalty-alpha")
        top_p_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Top-p (0.0 to disable sampling)")

        tts_downsampling_factor_slider = gr.Slider(minimum=1, maximum=6, value=1, step=1, label="TTS downsampling factor")
        tts_buffer_size_slider = gr.Slider(minimum=1, maximum=5, step=1, label="TTS buffer size",
                                           value=1 if self.args.tts_engine == "bark" else 4)
        tts_enhancement_dropdown = gr.Dropdown(label="TTS speech enhancement", choices=SpeechEnhancer.supported_models(), value="none")
        asr_max_buffer_size_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="ASR max buffer size")
        asr_logprob_threshold_slider = gr.Slider(minimum=-3.0, maximum=0.0, value=-0.7, step=0.05, label="ASR Log prob threshold")
        asr_no_speech_threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.05, label="ASR No speech threshold")

        asr_lang_dropdown = gr.Dropdown(choices=self.asr_handler.available_languages, label="ASR Language", value="English")

        if asr_lang_dropdown==self.asr_handler.AUTO_DETECT_LANG:
            asr_lang_dropdown=None

        #user and agent identities
        default_identities = Identity.default_identities()
        user_name_textbox = gr.Textbox(value=default_identities["S1"].name, label="User Name")
        user_age_textbox = gr.Textbox(value=default_identities["S1"].age, label="User Age")
        user_sex_dropdown = gr.Dropdown(
            choices=[default_identities["S1"].sex, "male", "female"], 
            value=default_identities["S1"].sex, label="User Gender"
        )
        agent_name_textbox = gr.Textbox(value=default_identities["S2"].name, label="Agent Name")
        agent_age_textbox = gr.Textbox(value=default_identities["S2"].age, label="Agent Age")
        agent_sex_dropdown = gr.Dropdown(
            choices=[default_identities["S2"].sex, "male", "female"], 
            value=default_identities["S2"].sex, label="Agent Gender"
        )
        agent_voice_dropdown = gr.Dropdown(
            choices=self.tts_handler.available_speakers, 
            value="en_speaker_8" if self.args.tts_engine == "bark" else "Voice 16",
            label="Agent Voice"
        )

        dialogue_chatbot = gr.Chatbot(label="Dialogue").style(color_map=("green", "pink"))
        reset_button = gr.Checkbox(value=True, label="Reset (holds agent in reset state until unchecked)",
                                   elem_id="reset_button")
        agent_starts_button = gr.Checkbox(value=True, label="Agent Starts (is the opening speaker)", elem_id="agent_starts_button")
        summary_textbox = gr.Textbox(
            label="Dialogue Summary Prompt", 
            value="S1 and S2 are talking about what's new in their lives."
        )
        opening_utt_textbox = gr.Textbox(label="Opening Utterance (agent only)", value="Hi! How are you?")
        
        state = gr.State({
            "dialogue": [], 
            "asr_config": ASRConfig(), 
            "agent_config": RealtimeAgentConfig(random_state=self.args.random_state, debug=self.args.debug),
            "tts_config": TTSConfig()
        })

        dialogue_interface = gr.Interface(
            fn=self.execute,
            inputs=[
                state,
                gr.Audio(source="microphone", streaming=True, label="ASR Input"),
                summary_textbox,
                opening_utt_textbox,
                reset_button,
                agent_starts_button,
                agent_interval_slider,
                similarity_threshold_slider,
                min_penalty_alpha_slider,
                max_penalty_alpha_slider,
                top_p_slider,
                tts_downsampling_factor_slider,
                tts_buffer_size_slider,
                tts_enhancement_dropdown,
                asr_max_buffer_size_slider,
                asr_model_size,
                asr_logprob_threshold_slider,
                asr_no_speech_threshold_slider,
                asr_lang_dropdown,
                user_name_textbox,
                user_age_textbox,
                user_sex_dropdown,
                agent_name_textbox,
                agent_age_textbox,
                agent_sex_dropdown,
                agent_voice_dropdown
                ], 
            outputs=[
                state,
                dialogue_chatbot,
                gr.Audio(elem_id="output_audio", label="TTS Output"),
                gr.HTML()
            ],
            live=True,
            allow_flagging='never',
            title=title,
            description=description
        )
        dialogue_interface.launch()

if __name__ == "__main__":
    parser = args_helpers.get_common_arg_parser()
    parser.add_argument("--tts-engine", type=str, default="fastspeech2", help="TTS engine to use")
    args = parser.parse_args()

    print("\nRunning with arguments:")
    print(args)
    print()

    interface = RealtimeAgentGradioInterface(args)
    interface.launch()