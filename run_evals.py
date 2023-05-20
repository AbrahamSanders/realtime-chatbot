from tqdm import tqdm, trange
import re
import torch
import multiprocessing as mp
import random
import ctypes
import pandas as pd
from datetime import datetime
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers.trainer_utils import set_seed

from realtime_chatbot.utils import args_helpers
from realtime_chatbot.realtime_agent import RealtimeAgent, RealtimeAgent_Resources, RealtimeAgentConfig
from realtime_chatbot.dynamic_contrastive import get_contrastive_search_override
from realtime_chatbot.utils.eval_utils import measure_common_responses
from simctg_evaluation import measure_repetition_and_diversity

speakers_in_prefix_regex = re.compile(r"S\d+(?= \(name:)")
speakers_in_transcript_regex = re.compile(r"S\d+(?=:)")
pauses_in_transcript_regex = re.compile(r"(?:<p> )?\((\d*?\.\d*?)\)")

SUPPORTED_DECODING_TYPES = ["greedy", "nucleus", "contrastive", "dynamic_contrastive", "contrastive_sampling", "dynamic_contrastive_sampling"]

def get_agent(args, device=None):
    agent = RealtimeAgent(
        resources=RealtimeAgent_Resources(
            modelpath=args.agent_modelpath, 
            device=device, 
            override_contrastive_search=False),
        config=RealtimeAgentConfig(
            prevent_special_token_generation=args.prevent_special_token_generation,
            add_special_pause_token=args.add_special_pause_token)
    )
    return agent

def set_generate_kwargs(agent, decoding_type):
    agent.generate_kwargs = {
        "pad_token_id": agent.resources.tokenizer.pad_token_id,
        "eos_token_id": agent.resources.tokenizer.eos_token_id
    }
    if decoding_type == "nucleus":
        agent.generate_kwargs["do_sample"] = True
        agent.generate_kwargs["top_p"] = 0.95
    if "contrastive" in decoding_type:
        agent.generate_kwargs["penalty_alpha"] = 0.6
        agent.generate_kwargs["top_k"] = 8
    if decoding_type == "dynamic_contrastive":
        agent.resources.model.contrastive_search = get_contrastive_search_override(agent.resources.model, 0.0, 1.0)
    # do_sample is not actually recognized by Generate for contrastive search. Setting it here to signal
    # to the evaluation routine that we want to generate multiple samples (the override supports sampling
    # but is not "officially" a sampling decoding type).
    if decoding_type == "contrastive_sampling":
        agent.generate_kwargs["do_sample"] = True
        agent.resources.model.contrastive_search = get_contrastive_search_override(agent.resources.model, 0.6, 0.6, sample_top_p=0.8)
    if decoding_type == "dynamic_contrastive_sampling":
        agent.generate_kwargs["do_sample"] = True
        agent.resources.model.contrastive_search = get_contrastive_search_override(agent.resources.model, 0.0, 1.0, sample_top_p=0.8)
        
def load_test_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.readlines()

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

def get_trp_examples(test_data, eval_mode, args):
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

def get_response_examples(test_data):
    response_examples = []
    transcript_marker = "Transcript: "
    for example in tqdm(test_data, desc="Preparing Response Examples"):
        prefix, transcript = example.split(transcript_marker)
        prefix += transcript_marker
        transcript_words = transcript.split()
        for i, word in enumerate(transcript_words):
            if re.match(speakers_in_transcript_regex, word):
                transcript_history = f"{prefix}{' '.join(transcript_words[:i+1])}"
                response_examples.append(transcript_history)

    # sort examples by length for greater batch efficiency
    response_examples.sort(key=len, reverse=True)
    return response_examples

def get_losses(agent, examples, batch_size, from_last_idx_of_token_id, show_progress=True):
    losses = []
    for start_index in trange(0, len(examples), batch_size, desc="Computing Losses", disable=not show_progress):
        examples_batch = examples[start_index : start_index+batch_size]
        inputs = agent.resources.tokenizer(
            examples_batch, padding=True, truncation=True, max_length=agent.tokenizer_max_length, return_tensors="pt"
        ).to(agent.resources.device)
        with torch.no_grad():
            logits = agent.resources.model(**inputs).logits
        labels = inputs.input_ids
        # From https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/opt/modeling_opt.py#L953
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.shape)
        actual_lengths = torch.unique_consecutive(inputs.attention_mask.nonzero(as_tuple=True)[0], return_counts=True)[1]
        for i, actual_length in enumerate(actual_lengths):
            last_idx_of_token = (shift_labels[i] == from_last_idx_of_token_id).nonzero(as_tuple=True)[0][-1]
            loss = token_losses[i, last_idx_of_token:actual_length-1].mean()
            losses.append(loss.item())

    return losses

def generate_for_predictions(agent, example, random_state, eos_token_id, max_new_tokens, multiple_samples=False):
    num_return_sequences = 5 if multiple_samples else 1
    num_generate_calls = 1
    do_sample = agent.generate_kwargs.get("do_sample", False)

    if do_sample and "penalty_alpha" in agent.generate_kwargs:
        # special case for dynamic contrastive search. Generate does not support do_sample=True or num_return_sequences > 1
        # for contrastive search, so we need to do multiple generate calls and pass do_sample=False 
        # (the override does the sampling regardless if sample_top_p is set)
        do_sample = False
        num_generate_calls = num_return_sequences
        num_return_sequences = 1

    if random_state is not None:
        set_seed(random_state)
    for _ in range(num_generate_calls):
        generated_preds = agent._generate(example, do_sample=do_sample, num_return_sequences=num_return_sequences, 
                                          eos_token_id=eos_token_id, max_new_tokens=max_new_tokens)
        if num_return_sequences == 1:
            generated_preds = [generated_preds]
        for pred in generated_preds:
            yield pred

def get_pause_predictions(agent, examples, decoding_type, random_state, show_progress=True):
    pause_at_1_preds = []
    pause_at_1_duration_errors = []
    pause_at_5_preds = []
    pause_at_5_duration_errors = []

    set_generate_kwargs(agent, decoding_type)
    multiple_samples = agent.generate_kwargs.get("do_sample", False)
    end_parens_token_id = agent.resources.tokenizer(")", add_special_tokens=False).input_ids[0]
    for example, target_duration in tqdm(examples, desc="Computing Pause Predictions", disable=not show_progress):
        predictions = generate_for_predictions(
            agent, example, random_state, eos_token_id=end_parens_token_id, max_new_tokens=7, multiple_samples=multiple_samples
        )
        pause_at_1 = 0
        pause_at_1_duration_error = None
        pause_at_5 = 0
        pause_at_5_duration_error = None
        for i, pred in enumerate(predictions):
            pause_match = re.match(agent.pause_regex, pred.lstrip())
            if pause_match:
                pause_duration = float(pause_match[1])
                pause_at_5 = 1
                if target_duration is not None:
                    pause_at_5_duration_error = abs(pause_duration - target_duration)
                if i == 0:
                    pause_at_1 = 1
                    if target_duration is not None:
                        pause_at_1_duration_error = abs(pause_duration - target_duration)
                break
        pause_at_1_preds.append(pause_at_1)
        if pause_at_1_duration_error is not None:
            pause_at_1_duration_errors.append(pause_at_1_duration_error)
        if multiple_samples:
            pause_at_5_preds.append(pause_at_5)
            if pause_at_5_duration_error is not None:
                pause_at_5_duration_errors.append(pause_at_5_duration_error)
    
    return pause_at_1_preds, pause_at_1_duration_errors, pause_at_5_preds, pause_at_5_duration_errors

def get_trp_predictions(agent, examples, decoding_type, random_state, show_progress=True):
    trp_at_1_preds = []
    trp_at_5_preds = []

    set_generate_kwargs(agent, decoding_type)
    multiple_samples = agent.generate_kwargs.get("do_sample", False)
    colon_token_id = agent.resources.tokenizer(":", add_special_tokens=False).input_ids[0]
    for example in tqdm(examples, desc="Computing TRP Predictions", disable=not show_progress):
        current_speaker = re.findall(speakers_in_transcript_regex, example)[-1]
        predictions = generate_for_predictions(
            agent, example, random_state, eos_token_id=colon_token_id, max_new_tokens=4, multiple_samples=multiple_samples
        )
        trp_at_1 = 0
        trp_at_5 = 0
        for i, pred in enumerate(predictions):
            pred_lstrip = pred.lstrip()
            if re.match(agent.any_identity_regex, pred_lstrip) and not pred_lstrip.startswith(f"{current_speaker}:"):
                trp_at_5 = 1
                if i == 0:
                    trp_at_1 = 1
                break
        trp_at_1_preds.append(trp_at_1)
        if multiple_samples:
            trp_at_5_preds.append(trp_at_5)
        
    return trp_at_1_preds, trp_at_5_preds

def get_response_predictions(agent, examples, decoding_type, random_state, show_progress=True):
    response_preds = []

    set_generate_kwargs(agent, decoding_type)
    turn_switch_token_id = agent.resources.tokenizer(" S", add_special_tokens=False).input_ids[0]
    for example in tqdm(examples, desc="Computing Response Predictions", disable=not show_progress):
        prediction = next(generate_for_predictions(agent, example, random_state, eos_token_id=turn_switch_token_id, max_new_tokens=60))
        prediction = prediction.rstrip("S").strip()
        response_preds.append(prediction)

    return response_preds

def setup_worker_pool(args):
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPUs detected. Using CPU for evaluation. This will be VERY slow.")
    ctx = mp.get_context("spawn")
    device_queue = ctx.SimpleQueue()
    for i in range(n_gpus):
        device_queue.put(i)
    decoding_type = ctx.Array(ctypes.c_char, 100)
    worker_pool = ctx.Pool(max(n_gpus, 1), initializer=eval_worker_init, initargs=(device_queue, decoding_type, args))
    return worker_pool, decoding_type

def eval_worker_init(device_queue, decoding_type, args):
    if not device_queue.empty():
        device_id = device_queue.get()
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    global worker_agent
    worker_agent = get_agent(args, device=device)
    global worker_decoding_type
    worker_decoding_type = decoding_type
    global worker_args
    worker_args = args

def eval_trp_losses_with_worker(batch):
    turn_switch_token_id = worker_agent.resources.tokenizer(" S", add_special_tokens=False).input_ids[0]
    return get_losses(worker_agent, batch, worker_args.batch_size, turn_switch_token_id, show_progress=False)

def eval_pause_losses_with_worker(batch):
    pause_token = "<p>" if worker_args.add_special_pause_token else " ("
    pause_token_id = worker_agent.resources.tokenizer(pause_token, add_special_tokens=False).input_ids[0]
    return get_losses(worker_agent, batch, worker_args.batch_size, pause_token_id, show_progress=False)

def eval_trp_preds_with_worker(batch):
    eval_decoding_type = worker_decoding_type.value.decode("utf-8")
    return get_trp_predictions(worker_agent, batch, eval_decoding_type, worker_args.random_state, show_progress=False)

def eval_pause_preds_with_worker(batch):
    eval_decoding_type = worker_decoding_type.value.decode("utf-8")
    return get_pause_predictions(worker_agent, batch, eval_decoding_type, worker_args.random_state, show_progress=False)

def eval_response_preds_with_worker(batch):
    eval_decoding_type = worker_decoding_type.value.decode("utf-8")
    return get_response_predictions(worker_agent, batch, eval_decoding_type, worker_args.random_state, show_progress=False)

def eval_ppl(worker_pool, examples, batch_size, batch_loss_eval_fn):
    losses = []
    batches = [examples[start:start+batch_size] for start in range(0, len(examples), batch_size)]
    with tqdm(total=len(batches), desc="Computing TRP Losses") as pbar:
        for res in worker_pool.imap_unordered(batch_loss_eval_fn, batches):
            losses.extend(res)
            pbar.update()

    ppl = torch.exp(torch.tensor(losses).mean()).item()
    return ppl

def eval_prediction(worker_pool, examples, targets, batch_pred_eval_fn):
    at_1_preds = []
    at_1_errors = []
    at_5_preds = []
    at_5_errors = []
    batches = [[example] for example in examples]
    with tqdm(total=len(batches), desc="Computing Predictions") as pbar:
        for res in worker_pool.imap(batch_pred_eval_fn, batches):
            if len(res) == 2:
                at_1_preds.extend(res[0])
                at_5_preds.extend(res[1])
            else:
                at_1_preds.extend(res[0])
                at_1_errors.extend(res[1])
                at_5_preds.extend(res[2])
                at_5_errors.extend(res[3])
            pbar.update()

    prec_at_1 = precision_score(targets, at_1_preds)
    recall_at_1 = recall_score(targets, at_1_preds)
    f1_at_1 = f1_score(targets, at_1_preds)
    error_at_1 = None
    if len(at_1_errors) > 0:
        error_at_1 = sum(at_1_errors) / len(at_1_errors)
    
    prec_at_5 = None
    recall_at_5 = None
    f1_at_5 = None
    error_at_5 = None
    if len(at_5_preds) > 0:
        prec_at_5 = precision_score(targets, at_5_preds)
        recall_at_5 = recall_score(targets, at_5_preds)
        f1_at_5 = f1_score(targets, at_5_preds)
        if len(at_5_errors) > 0:
            error_at_5 = sum(at_5_errors) / len(at_5_errors)

    return prec_at_1, recall_at_1, f1_at_1, error_at_1, prec_at_5, recall_at_5, f1_at_5, error_at_5

def eval_responses(worker_pool, examples):
    responses = []
    batches = [[example] for example in examples]
    with tqdm(total=len(batches), desc="Computing Response Predictions") as pbar:
        for res in worker_pool.imap_unordered(eval_response_preds_with_worker, batches):
            responses.extend(res)
            pbar.update()

    # we don't consider pauses in diversity calculation
    responses = [re.sub(pauses_in_transcript_regex, "", response) for response in responses]
    responses = [re.sub(" {2,}", " ", response) for response in responses]
    responses = [response for response in responses if response != ""]
    _, _, _, pred_div = measure_repetition_and_diversity(responses)
    pred_common = measure_common_responses(responses)
    return pred_div, pred_common

def print_and_append_to_results_dict(results_dict, eval_type, metric, result, do_print=True):
    key = f"{eval_type}_{metric}"
    if key not in results_dict:
        results_dict[key] = []
    results_dict[key].append(result)
    if do_print:
        print(f"{metric}: {result}")

def eval_and_print_ppl(worker_pool, args, eval_type, test_data, get_examples_fn, batch_loss_eval_fn, results_dict):
    if args.eval_type == "all" or args.eval_type == eval_type:
        print("-------------------------------------------------")
        print(f"-- Evaluating {eval_type}...")
        print("-------------------------------------------------")
        print()
        positive_examples, negative_examples = get_examples_fn(test_data, "ppl", args)
        print(f"# Positive {eval_type} examples: {len(positive_examples)}")
        print(f"# Negative {eval_type} examples: {len(negative_examples)}")
        print()
        positive_ppl = eval_ppl(worker_pool, positive_examples, args.batch_size, batch_loss_eval_fn)
        negative_ppl = eval_ppl(worker_pool, negative_examples, args.batch_size, batch_loss_eval_fn)
        print_and_append_to_results_dict(results_dict, eval_type, "pos", positive_ppl)
        print_and_append_to_results_dict(results_dict, eval_type, "neg", negative_ppl)
        print()

def eval_and_print_response_pred(worker_pool, decoding_type, args, test_data, results_dict):
    if args.eval_type == "all" or args.eval_type == "response_pred":
        eval_decoding_types = [args.decoding_type] if args.decoding_type != "all" else SUPPORTED_DECODING_TYPES
        for eval_decoding_type in eval_decoding_types:
            decoding_type.value = eval_decoding_type.encode("utf-8")
            print("-------------------------------------------------")
            print("-- Evaluating response_pred...")
            print(f"-- (decoding: {eval_decoding_type})")
            print("-------------------------------------------------")
            print()
            examples = get_response_examples(test_data)
            print(f"# response_pred examples: {len(examples)}")
            print()
            diversity, common = eval_responses(worker_pool, examples)
            print_and_append_to_results_dict(results_dict, "response_pred", "diversity", diversity)
            print_and_append_to_results_dict(results_dict, "response_pred", "common", common)
            print()
            
            # Add dummy @5 entry for sampling decodings
            if eval_decoding_type == "nucleus" or "sampling" in eval_decoding_type:
                print_and_append_to_results_dict(results_dict, "response_pred", "diversity", None, do_print=False)
                print_and_append_to_results_dict(results_dict, "response_pred", "common", None, do_print=False)

def eval_and_print_pred(worker_pool, decoding_type, args, eval_type, test_data, get_examples_fn, batch_pred_eval_fn, results_dict):
    if args.eval_type == "all" or args.eval_type == eval_type:
        eval_decoding_types = [args.decoding_type] if args.decoding_type != "all" else SUPPORTED_DECODING_TYPES
        for eval_decoding_type in eval_decoding_types:
            decoding_type.value = eval_decoding_type.encode("utf-8")
            print("-------------------------------------------------")
            print(f"-- Evaluating {eval_type}...")
            print(f"-- (decoding: {eval_decoding_type})")
            print("-------------------------------------------------")
            print()
            positive_examples, negative_examples = get_examples_fn(test_data, "pred", args)
            print(f"# Positive {eval_type} examples: {len(positive_examples)}")
            print(f"# Negative {eval_type} examples: {len(negative_examples)}")
            print()
            examples = positive_examples + negative_examples
            targets = [1] * len(positive_examples) + [0] * len(negative_examples)
            prec_at_1, recall_at_1, f1_at_1, error_at_1, prec_at_5, recall_at_5, f1_at_5, error_at_5 = eval_prediction(
                worker_pool, examples, targets, batch_pred_eval_fn
            )
            print("Results @1:")
            print_and_append_to_results_dict(results_dict, eval_type, "prec", prec_at_1)
            print_and_append_to_results_dict(results_dict, eval_type, "rec", recall_at_1)
            print_and_append_to_results_dict(results_dict, eval_type, "f1", f1_at_1)
            if error_at_1 is not None:
                print_and_append_to_results_dict(results_dict, eval_type, "error", error_at_1)
            print()
            if prec_at_5 is not None:
                print("Results @5:")
                print_and_append_to_results_dict(results_dict, eval_type, "prec", prec_at_5)
                print_and_append_to_results_dict(results_dict, eval_type, "rec", recall_at_5)
                print_and_append_to_results_dict(results_dict, eval_type, "f1", f1_at_5)
                if error_at_5 is not None:
                    print_and_append_to_results_dict(results_dict, eval_type, "error", error_at_5)
                print()

def get_pred_results_df_index(args):
    eval_decoding_types = [args.decoding_type] if args.decoding_type != "all" else SUPPORTED_DECODING_TYPES
    index = []
    for eval_decoding_type in eval_decoding_types:
        if eval_decoding_type == "nucleus" or "sampling" in eval_decoding_type:
            index.append(f"{eval_decoding_type} @ 1")
            index.append(f"{eval_decoding_type} @ 5")
        else:
            index.append(eval_decoding_type)
    return index

if __name__ == "__main__":
    parser = args_helpers.get_common_arg_parser()
    parser.add_argument("--test-data", default="data/dataset_test_dyads_original_pauses.txt")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--data-random-state", type=int, default=42)
    parser.add_argument("--eval-type", choices=["all", "trp_ppl", "trp_pred", "pause_ppl", "pause_pred", "response_pred"], default="all")
    parser.add_argument("--decoding-type", choices=["all"] + SUPPORTED_DECODING_TYPES, default="all")
    args = parser.parse_args()

    print("\nRunning with arguments:")
    print(args)
    print()

    if args.random_state is None:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("WARNING: random_state not set. \n"
              "Results will not be reproducible. \n"
              "Are you sure you want to do this???")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()

    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    test_data = load_test_data(args.test_data)
    if args.num_examples > 0:
        random.seed(args.data_random_state)
        test_data = random.sample(test_data, args.num_examples)

    ppl_results_dict = {}
    pred_results_dict = {}
    worker_pool, decoding_type = setup_worker_pool(args)
    with worker_pool:
        # Turn-taking Evals
        eval_and_print_ppl(worker_pool, args, "trp_ppl", test_data, get_trp_examples, eval_trp_losses_with_worker, ppl_results_dict)
        eval_and_print_pred(worker_pool, decoding_type, args, "trp_pred", test_data, get_trp_examples, eval_trp_preds_with_worker, pred_results_dict)

        # Pausing Evals
        eval_and_print_ppl(worker_pool, args, "pause_ppl", test_data, get_pause_examples, eval_pause_losses_with_worker, ppl_results_dict)
        eval_and_print_pred(worker_pool, decoding_type, args, "pause_pred", test_data, get_pause_examples, eval_pause_preds_with_worker, pred_results_dict)

        # Response Evals
        eval_and_print_response_pred(worker_pool, decoding_type, args, test_data, pred_results_dict)

    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    runtime_minutes = int((end_time - start_time).total_seconds() / 60)
    print(f"Total time: {runtime_minutes} minutes.")

    if ppl_results_dict:
        ppl_results_df = pd.DataFrame.from_dict(ppl_results_dict)
        print(ppl_results_df)
        ppl_results_df.to_csv("evals_output_ppl.csv", index=False)

    if pred_results_dict:
        pred_results_df = pd.DataFrame.from_dict(pred_results_dict)
        pred_results_df.index = get_pred_results_df_index(args)
        print(pred_results_df)
        pred_results_df.to_csv("evals_output_pred.csv")