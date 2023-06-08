import argparse

def get_common_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-modelpath", default="AbrahamSanders/opt-2.7b-realtime-chat-v2", 
                        help="Path to to the HuggingFace transformers model to use for the agent. (default: %(default)s)")
    parser.add_argument("--random-state", type=int, default=None,
                        help="Random seed for model reproducibility. (default: %(default)s)")
    parser.add_argument("--prevent-special-token-generation", action="store_true",
                        help="Use with base OPT model for zero-shot inference. (default: %(default)s)")
    parser.add_argument("--add-special-pause-token", action="store_true",
                        help="Add special <p> token when incrementing pauses. (default: %(default)s)")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages. (default: %(default)s)")
    return parser