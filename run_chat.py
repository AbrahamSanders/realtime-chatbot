from realtime_chatbot.realtime_agent import RealtimeAgentMultiprocessing, RealtimeAgentConfig
from sshkeyboard import listen_keyboard, stop_listening
from threading import Thread
from queue import SimpleQueue
from time import sleep

from realtime_chatbot.utils import queue_helpers, args_helpers
from realtime_chatbot.identity import Identity

class KeyboardListener:
    def __init__(self):
        self.input_buffer = ""
        self.input_queue = SimpleQueue()
        self.listen_thread = Thread(target=self.listen, daemon=True) 
        self.listen_thread.start()

    def listen(self):
        listen_keyboard(on_press=self.on_press, delay_second_char=0.05, lower=False)

    def on_press(self, key):
        if key in ["space", "enter"]:
            self.input_queue.put(self.input_buffer)
            self.input_buffer = ""
        elif key == "backspace":
            if len(self.input_buffer) > 0:
                self.input_buffer = self.input_buffer[:-1]
        else:
            self.input_buffer += key

    def next_input(self):
        return queue_helpers.join_queue(self.input_queue)

def configure_identities():
    identities = Identity.default_identities()
    for identity, info in identities.items():
        name = input(f"What is {identity}'s name? ")
        age = input(f"How old is {identity}? ")
        sex = input(f"What is {identity}'s gender (male, female, unknown)? ")
        info.name = name if name else info.name
        info.age = age if age else info.age
        info.sex = sex if sex else info.sex
    return identities

def main():
    parser = args_helpers.get_common_arg_parser()
    args = parser.parse_args()

    print("\nRunning with arguments:")
    print(args)
    print()

    identities = configure_identities()
    config = RealtimeAgentConfig(identities=identities, random_state=args.random_state)
    agent = RealtimeAgentMultiprocessing(config=config, modelpath=args.agent_modelpath)
    listener = KeyboardListener()
    user_speaking = None
    print("\n\n>>>Running<<<\n\n")
    while True:
        # User input (if any)
        next_input = listener.next_input()
        if next_input:
            if next_input.endswith("-exit"):
                stop_listening()
                break
            if next_input.endswith("-reset"):
                agent.queue_reset()
                print("\n\n>>>Reset<<<\n")
                continue
            agent.queue_input(next_input)
            if user_speaking is None or not user_speaking:
                user_speaking = True
                print(f"\n{config.user_identity}:", end="", flush=True)
            print(f" {next_input}", end="", flush=True)

        # Agent output (if any)
        next_output = agent.next_output()
        if next_output:
            if user_speaking is None or user_speaking:
                user_speaking = False
                print(f"\n{config.agent_identity}:", end="", flush=True)
            print(next_output, end="", flush=True)

        # Brief sleep to avoid tight loop
        sleep(0.05)

if __name__ == "__main__":
    main()