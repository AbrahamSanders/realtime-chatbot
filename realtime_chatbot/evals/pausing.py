from tqdm import tqdm
import re
import torch

from . import common as cm
from .data_processing import speaker_regex, pause_regex

def get_ppl_pause_examples(test_data, args):
    positive_pause_examples = []
    negative_pause_examples = []
    transcript_marker = "Transcript: "
    for example in tqdm(test_data, desc="Preparing PPL (Pause) Examples"):
        prefix, transcript = example.split(transcript_marker)
        prefix += transcript_marker
        pauses_in_transcript = re.findall(pause_regex, transcript)
        # If there are no pauses at all in the transcript, the transcriber likely did not annotate pauses
        # and it doesn't make sense to evaluate this example.
        if len(pauses_in_transcript) == 0:
            continue
        transcript_words = transcript.split()
        last_word = ""
        for i, word in enumerate(transcript_words):
            word_pause_match = re.match(pause_regex, word)
            transcript_history = f"{prefix}{' '.join(transcript_words[:i])}"
            # If we are at pause, this is a positive pause example.
            # If we are not at a pause, this is a negative pause example.
            if word_pause_match:
                positive_pause_examples.append(f"{transcript_history} {word_pause_match[0]}")
            # Do not use the beginning or end of a turn as a negative pause example because these are common
            # places for speakers to pause, even if not explicitly annotated.
            elif not re.match(speaker_regex, word) and not re.match(speaker_regex, last_word):
                for pause_duration in (0.2, 0.5, 1.0):
                    pause_prefix = "<p> " if args.add_special_pause_token else ""
                    pause = f"{pause_prefix}({pause_duration:.1f})"
                    negative_pause_examples.append(f"{transcript_history} {pause}")
            last_word = word
    
    # sort examples by length for greater batch efficiency
    positive_pause_examples.sort(key=len, reverse=True)
    negative_pause_examples.sort(key=len, reverse=True)
    return positive_pause_examples, negative_pause_examples

def get_pause_losses_with_worker(batch):
    pause_token = "<p>" if cm.worker_args.add_special_pause_token else " ("
    return cm.get_losses(cm.worker_agent, batch, cm.worker_args.batch_size, from_last_idx_of_token=pause_token, show_progress=False)

def eval_pause_ppl(worker_pool, args, test_data, batch_size):
    positive_examples, negative_examples = get_ppl_pause_examples(test_data, args)
    print(f"# Positive PPL (Pause) examples: {len(positive_examples)}")
    print(f"# Negative PPL (Pause) examples: {len(negative_examples)}")
    print()

    positive_losses = cm.get_model_output_with_worker_pool(worker_pool, positive_examples, batch_size, get_pause_losses_with_worker)
    positive_ppl = torch.exp(torch.tensor(positive_losses).mean()).item()

    negative_losses = cm.get_model_output_with_worker_pool(worker_pool, negative_examples, batch_size, get_pause_losses_with_worker)
    negative_ppl = torch.exp(torch.tensor(negative_losses).mean()).item()

    return [
        ("pos", positive_ppl),
        ("neg", negative_ppl)
    ]