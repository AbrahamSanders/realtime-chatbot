import argparse

def get_common_arg_parser():
    parser = argparse.ArgumentParser("Run chat from a terminal.")
    parser.add_argument("--agent-modelpath", default="AbrahamSanders/opt-2.7b-realtime-chat", 
                        help="Path to to the HuggingFace transformers model to use for the agent. (default: %(default)s)")
    parser.add_argument("--random-state", type=int, default=None,
                        help="Random seed for model reproducibility. (default: %(default)s)")
    return parser