from tqdm import tqdm
import re
import torch

from . import common as cm
from .data_processing import speaker_regex

speakers_in_prefix_regex = re.compile(r"S\d+(?= \(name:)")

def get_ppl_trp_examples(test_data):
    positive_trp_examples = []
    negative_trp_examples = []
    transcript_marker = "Transcript: "
    for example in tqdm(test_data, desc="Preparing PPL (TRP) Examples"):
        prefix, transcript = example.split(transcript_marker)
        prefix += transcript_marker
        speakers = re.findall(speakers_in_prefix_regex, prefix)
        speakers_in_transcript = set(re.findall(speaker_regex, transcript))
        # If there are less than 2 active speakers in the transcript, no turn-taking takes place and it doesn't
        # make sense to evaluate this example.
        if len(speakers_in_transcript) < 2:
            continue
        transcript_words = transcript.split()
        current_speaker = None
        for i, word in enumerate(transcript_words):
            word_speaker_identity_match = re.match(speaker_regex, word)
            if current_speaker is not None:
                transcript_history = f"{prefix}{' '.join(transcript_words[:i])}"
                # If we are at a turn switch to another speaker, this is a positive TRP example.
                # If we are not at a turn switch, this is a negative TRP example.
                if word_speaker_identity_match:
                    # Positive TRP only makes sense for a turn-switch to another speaker.
                    # If we are at a turn switch to the same speaker (repeated speaker identity for a new utterance), skip it.
                    if word_speaker_identity_match[0] != current_speaker:
                        positive_trp_examples.append(f"{transcript_history} {word_speaker_identity_match[0]}:")
                else:
                    for speaker in speakers:
                        if speaker != current_speaker:
                            negative_trp_examples.append(f"{transcript_history} {speaker}:")

            if word_speaker_identity_match:
                current_speaker = word_speaker_identity_match[0]

    # sort examples by length for greater batch efficiency
    positive_trp_examples.sort(key=len, reverse=True)
    negative_trp_examples.sort(key=len, reverse=True)
    return positive_trp_examples, negative_trp_examples

def get_trp_losses_with_worker(batch):
    return cm.get_losses(cm.worker_agent, batch, cm.worker_args.batch_size, from_last_idx_of_token=" S", show_progress=False)

def eval_trp_ppl(worker_pool, args, test_data, batch_size):
    positive_examples, negative_examples = get_ppl_trp_examples(test_data)
    print(f"# Positive PPL (TRP) examples: {len(positive_examples)}")
    print(f"# Negative PPL (TRP) examples: {len(negative_examples)}")
    print()

    positive_losses = cm.get_model_output_with_worker_pool(worker_pool, positive_examples, batch_size, get_trp_losses_with_worker)
    positive_ppl = torch.exp(torch.tensor(positive_losses).mean()).item()

    negative_losses = cm.get_model_output_with_worker_pool(worker_pool, negative_examples, batch_size, get_trp_losses_with_worker)
    negative_ppl = torch.exp(torch.tensor(negative_losses).mean()).item()

    return [
        ("pos", positive_ppl),
        ("neg", negative_ppl)
    ]