from tqdm import tqdm
import re
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from . import common as cm

speakers_in_prefix_regex = re.compile(r"S\d+(?= \(name:)")
speakers_in_transcript_regex = re.compile(r"S\d+(?=:)")

def get_trp_examples(test_data, eval_mode):
    if eval_mode not in ["ppl", "pred"]:
        raise ValueError(f"eval_mode must be 'ppl' or 'pred'. Got {eval_mode}.")

    positive_trp_examples = []
    negative_trp_examples = []
    transcript_marker = "Transcript: "
    for example in tqdm(test_data, desc=f"Preparing TRP Examples ({eval_mode})"):
        prefix, transcript = example.split(transcript_marker)
        prefix += transcript_marker
        speakers = re.findall(speakers_in_prefix_regex, prefix)
        speakers_in_transcript = set(re.findall(speakers_in_transcript_regex, transcript))
        # If there are less than 2 active speakers in the transcript, no turn-taking takes place and it doesn't
        # make sense to evaluate this example.
        if len(speakers_in_transcript) < 2:
            continue
        transcript_words = transcript.split()
        current_speaker = None
        for i, word in enumerate(transcript_words):
            word_speaker_identity_match = re.match(speakers_in_transcript_regex, word)
            if current_speaker is not None:
                transcript_history = f"{prefix}{' '.join(transcript_words[:i])}"
                # If we are at a turn switch to another speaker, this is a positive TRP example.
                # If we are not at a turn switch, this is a negative TRP example.
                if word_speaker_identity_match:
                    # Positive TRP only makes sense for a turn-switch to another speaker.
                    # If we are at a turn switch to the same speaker (repeated speaker identity for a new utterance), skip it.
                    if word_speaker_identity_match[0] != current_speaker:
                        if eval_mode == "ppl":
                            positive_trp_examples.append(f"{transcript_history} {word_speaker_identity_match[0]}:")
                        else:
                            positive_trp_examples.append(transcript_history)
                else:
                    if eval_mode == "ppl":
                        for speaker in speakers:
                            if speaker != current_speaker:
                                negative_trp_examples.append(f"{transcript_history} {speaker}:")
                    else:
                        negative_trp_examples.append(transcript_history)

            if word_speaker_identity_match:
                current_speaker = word_speaker_identity_match[0]

    # sort examples by length for greater batch efficiency
    positive_trp_examples.sort(key=len, reverse=True)
    negative_trp_examples.sort(key=len, reverse=True)
    return positive_trp_examples, negative_trp_examples

def get_trp_losses_with_worker(batch):
    return cm.get_losses(cm.worker_agent, batch, cm.worker_args.batch_size, from_last_idx_of_token=" S", show_progress=False)

def get_trp_preds_with_worker(batch):
    eval_decoding_type = cm.worker_decoding_type.value.decode("utf-8")
    batch_size = cm.get_batch_size(cm.worker_args, eval_decoding_type)
    return cm.get_predictions(cm.worker_agent, batch, batch_size, eos_token=":", strip_eos_token=False, 
                              max_new_tokens=4, decoding_type=eval_decoding_type, show_progress=False)

def eval_trp_ppl(worker_pool, args, test_data, batch_size):
    positive_examples, negative_examples = get_trp_examples(test_data, "ppl")
    print(f"# Positive TRP examples: {len(positive_examples)}")
    print(f"# Negative TRP examples: {len(negative_examples)}")
    print()

    positive_losses = cm.get_model_output_with_worker_pool(worker_pool, positive_examples, batch_size, get_trp_losses_with_worker)
    positive_ppl = torch.exp(torch.tensor(positive_losses).mean()).item()

    negative_losses = cm.get_model_output_with_worker_pool(worker_pool, negative_examples, batch_size, get_trp_losses_with_worker)
    negative_ppl = torch.exp(torch.tensor(negative_losses).mean()).item()

    return [
        ("pos", positive_ppl),
        ("neg", negative_ppl)
    ]

def eval_trp_pred(worker_pool, args, test_data, batch_size):
    positive_examples, negative_examples = get_trp_examples(test_data, "pred")
    print(f"# Positive TRP examples: {len(positive_examples)}")
    print(f"# Negative TRP examples: {len(negative_examples)}")
    print()

    examples = positive_examples + negative_examples
    targets = [1] * len(positive_examples) + [0] * len(negative_examples)
    predictions = cm.get_model_output_with_worker_pool(worker_pool, examples, batch_size, get_trp_preds_with_worker)

    trp_preds = []
    for example, pred in zip(examples, predictions):
        current_speaker = re.findall(speakers_in_transcript_regex, example)[-1]
        pred_lstrip = pred.lstrip()
        trp = 0
        if re.match(speakers_in_transcript_regex, pred_lstrip) and not pred_lstrip.startswith(f"{current_speaker}:"):
            trp = 1
        trp_preds.append(trp)

    prec = precision_score(targets, trp_preds)
    recall = recall_score(targets, trp_preds)
    f1 = f1_score(targets, trp_preds)

    return [
        ("prec", prec),
        ("rec", recall),
        ("f1", f1)
    ]