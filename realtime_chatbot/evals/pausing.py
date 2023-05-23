from tqdm import tqdm
import re
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from . import common as cm
from .turn_taking import speakers_in_transcript_regex

pauses_in_transcript_regex = re.compile(r"(?:<p> )?\((\d*?\.\d*?)\)")

def get_pause_examples(test_data, eval_mode, args):
    if eval_mode not in ["ppl", "pred"]:
        raise ValueError(f"eval_mode must be 'ppl' or 'pred'. Got {eval_mode}.")
    
    positive_pause_examples = []
    negative_pause_examples = []
    transcript_marker = "Transcript: "
    for example in tqdm(test_data, desc=f"Preparing Pause Examples ({eval_mode})"):
        prefix, transcript = example.split(transcript_marker)
        prefix += transcript_marker
        pauses_in_transcript = re.findall(pauses_in_transcript_regex, transcript)
        # If there are no pauses at all in the transcript, the transcriber likely did not annotate pauses
        # and it doesn't make sense to evaluate this example.
        if len(pauses_in_transcript) == 0:
            continue
        transcript_words = transcript.split()
        last_word = ""
        for i, word in enumerate(transcript_words):
            word_pause_match = re.match(pauses_in_transcript_regex, word)
            transcript_history = f"{prefix}{' '.join(transcript_words[:i])}"
            # If we are at pause, this is a positive pause example.
            # If we are not at a pause, this is a negative pause example.
            if word_pause_match:
                if eval_mode == "ppl":
                    positive_pause_examples.append(f"{transcript_history} {word_pause_match[0]}")
                else:
                    positive_pause_examples.append((transcript_history, float(word_pause_match[1])))
            # Do not use the beginning or end of a turn as a negative pause example because these are common
            # places for speakers to pause, even if not explicitly annotated.
            elif not re.match(speakers_in_transcript_regex, word) and not re.match(speakers_in_transcript_regex, last_word):
                if eval_mode == "ppl":
                    for pause_duration in (0.2, 0.5, 1.0):
                        pause_prefix = "<p> " if args.add_special_pause_token else ""
                        pause = f"{pause_prefix}({pause_duration:.1f})"
                        negative_pause_examples.append(f"{transcript_history} {pause}")
                else:
                    negative_pause_examples.append((transcript_history, None))
            last_word = word
    
    # sort examples by length for greater batch efficiency
    if eval_mode == "ppl":
        positive_pause_examples.sort(key=len, reverse=True)
        negative_pause_examples.sort(key=len, reverse=True)
    else:
        positive_pause_examples.sort(key=lambda x: len(x[0]), reverse=True)
        negative_pause_examples.sort(key=lambda x: len(x[0]), reverse=True)
    return positive_pause_examples, negative_pause_examples

def get_pause_losses_with_worker(batch):
    pause_token = "<p>" if cm.worker_args.add_special_pause_token else " ("
    return cm.get_losses(cm.worker_agent, batch, cm.worker_args.batch_size, from_last_idx_of_token=pause_token, show_progress=False)

def get_pause_preds_with_worker(batch):
    eval_decoding_type = cm.worker_decoding_type.value.decode("utf-8")
    batch_size = cm.get_batch_size(cm.worker_args, eval_decoding_type)
    return cm.get_predictions(cm.worker_agent, batch, batch_size, eos_token=")", strip_eos_token=False, 
                              max_new_tokens=7, decoding_type=eval_decoding_type, show_progress=False)

def eval_pause_ppl(worker_pool, args, test_data, batch_size):
    positive_examples, negative_examples = get_pause_examples(test_data, "ppl", args)
    print(f"# Positive Pause examples: {len(positive_examples)}")
    print(f"# Negative Pause examples: {len(negative_examples)}")
    print()

    positive_losses = cm.get_model_output_with_worker_pool(worker_pool, positive_examples, batch_size, get_pause_losses_with_worker)
    positive_ppl = torch.exp(torch.tensor(positive_losses).mean()).item()

    negative_losses = cm.get_model_output_with_worker_pool(worker_pool, negative_examples, batch_size, get_pause_losses_with_worker)
    negative_ppl = torch.exp(torch.tensor(negative_losses).mean()).item()

    return [
        ("pos", positive_ppl),
        ("neg", negative_ppl)
    ]

def eval_pause_pred(worker_pool, args, test_data, batch_size):
    positive_examples, negative_examples = get_pause_examples(test_data, "pred", args)
    print(f"# Positive Pause examples: {len(positive_examples)}")
    print(f"# Negative Pause examples: {len(negative_examples)}")
    print()

    examples = positive_examples + negative_examples
    examples, duration_targets = zip(*examples)
    targets = [1] * len(positive_examples) + [0] * len(negative_examples)
    predictions = cm.get_model_output_with_worker_pool(worker_pool, examples, batch_size, get_pause_preds_with_worker)

    pause_preds = []
    pause_duration_errors = []
    for pred, target_duration in zip(predictions, duration_targets):
        pause_match = re.match(pauses_in_transcript_regex, pred.lstrip())
        pause = 0
        pause_duration_error = None
        if pause_match:
            pause = 1
            if target_duration is not None:
                pause_duration = float(pause_match[1])
                pause_duration_error = abs(pause_duration - target_duration)
        pause_preds.append(pause)
        if pause_duration_error is not None:
            pause_duration_errors.append(pause_duration_error)

    prec = precision_score(targets, pause_preds)
    recall = recall_score(targets, pause_preds)
    f1 = f1_score(targets, pause_preds)
    error = None
    if len(pause_duration_errors) > 0:
        error = sum(pause_duration_errors) / len(pause_duration_errors)

    return [
        ("prec", prec),
        ("rec", recall),
        ("f1", f1),
        ("error", error)
    ]