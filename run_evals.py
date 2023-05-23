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
            random_state=args.random_state,
            prevent_special_token_generation=args.prevent_special_token_generation,
            add_special_pause_token=args.add_special_pause_token)
    )
    agent.resources.model.contrastive_search_original = agent.resources.model.contrastive_search
    return agent

def get_batch_size(args, decoding_type):
    if decoding_type is not None and "contrastive" in decoding_type:
        return args.contrastive_batch_size
    else:
        return args.batch_size

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
    if decoding_type == "contrastive":
        agent.resources.model.contrastive_search = agent.resources.model.contrastive_search_original
    if decoding_type == "dynamic_contrastive":
        agent.resources.model.contrastive_search = get_contrastive_search_override(agent.resources.model, 0.0, 1.0)
    if decoding_type == "contrastive_sampling":
        agent.resources.model.contrastive_search = get_contrastive_search_override(agent.resources.model, 0.6, 0.6, sample_top_p=0.8)
    if decoding_type == "dynamic_contrastive_sampling":
        agent.resources.model.contrastive_search = get_contrastive_search_override(agent.resources.model, 0.0, 1.0, sample_top_p=0.8)
        
def load_test_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.readlines()

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

def get_losses(agent, examples, batch_size, from_last_idx_of_token, show_progress=True):
    losses = []
    from_last_idx_of_token_id = agent.resources.tokenizer(from_last_idx_of_token, add_special_tokens=False).input_ids[0]
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

def get_predictions(agent, examples, batch_size, eos_token, strip_eos_token, max_new_tokens, decoding_type, show_progress=True):
    predictions = []
    set_generate_kwargs(agent, decoding_type)
    eos_token_id = agent.resources.tokenizer(eos_token, add_special_tokens=False).input_ids[0]
    for start_index in trange(0, len(examples), batch_size, desc="Computing Predictions", disable=not show_progress):
        examples_batch = examples[start_index : start_index+batch_size]
        batch_preds = agent._generate(examples_batch, eos_token_id=eos_token_id, max_new_tokens=max_new_tokens)
        if strip_eos_token:
            batch_preds = [pred.rstrip(eos_token) for pred in batch_preds]
        predictions.extend(batch_preds)

    return predictions

def setup_worker_pool(args):
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPUs detected. Using CPU for evaluation. This will be VERY slow.")
    ctx = mp.get_context("spawn")
    device_queue = ctx.SimpleQueue()
    for i in range(n_gpus):
        device_queue.put(i)
    decoding_type = ctx.Array(ctypes.c_char, 100)
    worker_pool = ctx.Pool(max(n_gpus, 1), initializer=worker_init, initargs=(device_queue, decoding_type, args))
    return worker_pool, decoding_type

def worker_init(device_queue, decoding_type, args):
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

def get_trp_losses_with_worker(batch):
    return get_losses(worker_agent, batch, worker_args.batch_size, from_last_idx_of_token=" S", show_progress=False)

def get_pause_losses_with_worker(batch):
    pause_token = "<p>" if worker_args.add_special_pause_token else " ("
    return get_losses(worker_agent, batch, worker_args.batch_size, from_last_idx_of_token=pause_token, show_progress=False)

def get_trp_preds_with_worker(batch):
    eval_decoding_type = worker_decoding_type.value.decode("utf-8")
    batch_size = get_batch_size(worker_args, eval_decoding_type)
    return get_predictions(worker_agent, batch, batch_size, eos_token=":", strip_eos_token=False, 
                           max_new_tokens=4, decoding_type=eval_decoding_type, show_progress=False)

def get_pause_preds_with_worker(batch):
    eval_decoding_type = worker_decoding_type.value.decode("utf-8")
    batch_size = get_batch_size(worker_args, eval_decoding_type)
    return get_predictions(worker_agent, batch, batch_size, eos_token=")", strip_eos_token=False, 
                           max_new_tokens=7, decoding_type=eval_decoding_type, show_progress=False)

def get_response_preds_with_worker(batch):
    eval_decoding_type = worker_decoding_type.value.decode("utf-8")
    batch_size = get_batch_size(worker_args, eval_decoding_type)
    return get_predictions(worker_agent, batch, batch_size, eos_token=" S", strip_eos_token=True, 
                           max_new_tokens=60, decoding_type=eval_decoding_type, show_progress=False)

def get_model_output_with_worker_pool(worker_pool, examples, batch_size, batch_get_output_fn):
    worker_outputs = []
    batches = [examples[start:start+batch_size] for start in range(0, len(examples), batch_size)]
    with tqdm(total=len(examples), desc="Computing Model Output") as pbar:
        for res in worker_pool.imap(batch_get_output_fn, batches):
            worker_outputs.extend(res)
            pbar.update(len(res))
    return worker_outputs

def print_and_append_to_results_dict(results_dict, eval_type, metric, result):
    key = f"{eval_type}_{metric}"
    if key not in results_dict:
        results_dict[key] = []
    results_dict[key].append(result)
    print(f"{metric}: {result}")

def eval_and_print(eval_type, eval_fn, decoding_type, worker_pool, args, test_data, results_dict):
    if args.eval_type == "all" or args.eval_type == eval_type:
        if decoding_type is not None:
            eval_decoding_types = [args.decoding_type] if args.decoding_type != "all" else SUPPORTED_DECODING_TYPES
        else:
            eval_decoding_types = [None]

        for eval_decoding_type in eval_decoding_types:
            print("-------------------------------------------------")
            print(f"-- Evaluating {eval_type}...")
            if eval_decoding_type is not None:
                print(f"-- (decoding: {eval_decoding_type})")
                decoding_type.value = eval_decoding_type.encode("utf-8")
            print("-------------------------------------------------")
            print()
            batch_size = get_batch_size(args, eval_decoding_type)
            results = eval_fn(worker_pool, args, test_data, batch_size)
            for metric, result in results:
                print_and_append_to_results_dict(results_dict, eval_type, metric, result)
            print()

def eval_trp_ppl(worker_pool, args, test_data, batch_size):
    positive_examples, negative_examples = get_trp_examples(test_data, "ppl")
    print(f"# Positive TRP examples: {len(positive_examples)}")
    print(f"# Negative TRP examples: {len(negative_examples)}")
    print()

    positive_losses = get_model_output_with_worker_pool(worker_pool, positive_examples, batch_size, get_trp_losses_with_worker)
    positive_ppl = torch.exp(torch.tensor(positive_losses).mean()).item()

    negative_losses = get_model_output_with_worker_pool(worker_pool, negative_examples, batch_size, get_trp_losses_with_worker)
    negative_ppl = torch.exp(torch.tensor(negative_losses).mean()).item()

    return [
        ("pos", positive_ppl),
        ("neg", negative_ppl)
    ]

def eval_pause_ppl(worker_pool, args, test_data, batch_size):
    positive_examples, negative_examples = get_pause_examples(test_data, "ppl", args)
    print(f"# Positive Pause examples: {len(positive_examples)}")
    print(f"# Negative Pause examples: {len(negative_examples)}")
    print()

    positive_losses = get_model_output_with_worker_pool(worker_pool, positive_examples, batch_size, get_pause_losses_with_worker)
    positive_ppl = torch.exp(torch.tensor(positive_losses).mean()).item()

    negative_losses = get_model_output_with_worker_pool(worker_pool, negative_examples, batch_size, get_pause_losses_with_worker)
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
    predictions = get_model_output_with_worker_pool(worker_pool, examples, batch_size, get_trp_preds_with_worker)

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

def eval_pause_pred(worker_pool, args, test_data, batch_size):
    positive_examples, negative_examples = get_pause_examples(test_data, "pred", args)
    print(f"# Positive Pause examples: {len(positive_examples)}")
    print(f"# Negative Pause examples: {len(negative_examples)}")
    print()

    examples = positive_examples + negative_examples
    examples, duration_targets = zip(*examples)
    targets = [1] * len(positive_examples) + [0] * len(negative_examples)
    predictions = get_model_output_with_worker_pool(worker_pool, examples, batch_size, get_pause_preds_with_worker)

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

def eval_response_pred(worker_pool, args, test_data, batch_size):
    examples = get_response_examples(test_data)
    print(f"# Response Prediction examples: {len(examples)}")
    print()

    predictions = get_model_output_with_worker_pool(worker_pool, examples, batch_size, get_response_preds_with_worker)

    # we don't consider pauses in diversity calculation
    predictions = [re.sub(pauses_in_transcript_regex, "", response) for response in predictions]
    predictions = [re.sub(" {2,}", " ", response) for response in predictions]
    predictions = [response.strip() for response in predictions]
    predictions = [response for response in predictions if response != ""]

    _, _, _, diversity = measure_repetition_and_diversity(predictions)
    common = measure_common_responses(predictions)

    return [
        ("diversity", diversity),
        ("common", common)
    ]

if __name__ == "__main__":
    parser = args_helpers.get_common_arg_parser()
    parser.add_argument("--test-data", default="data/dataset_test_dyads_original_pauses.txt")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--contrastive-batch-size", type=int, default=5)
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--data-random-state", type=int, default=42)
    parser.add_argument("--eval-type", choices=["all", "trp_ppl", "trp_pred", "pause_ppl", "pause_pred", "response_pred"], default="all")
    parser.add_argument("--decoding-type", choices=["all"] + SUPPORTED_DECODING_TYPES, default="all")
    args = parser.parse_args()

    if args.random_state is None:
        print("\nrandom_state not set. Setting to 42...")
        args.random_state = 42

    print("\nRunning with arguments:")
    print(args)
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
        eval_and_print("trp_ppl", eval_trp_ppl, None, worker_pool, args, test_data, ppl_results_dict)
        eval_and_print("trp_pred", eval_trp_pred, decoding_type, worker_pool, args, test_data, pred_results_dict)

        # Pausing Evals
        eval_and_print("pause_ppl", eval_pause_ppl, None, worker_pool, args, test_data, ppl_results_dict)
        eval_and_print("pause_pred", eval_pause_pred, decoding_type, worker_pool, args, test_data, pred_results_dict)

        # Response Evals
        eval_and_print("response_pred", eval_response_pred, decoding_type, worker_pool, args, test_data, pred_results_dict)

    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    runtime_minutes = int((end_time - start_time).total_seconds() / 60)
    print(f"Total time: {runtime_minutes} minutes.")

    if ppl_results_dict:
        ppl_results_df = pd.DataFrame.from_dict(ppl_results_dict)
        print(ppl_results_df)
        print()
        ppl_results_df.to_csv(f"evals_output_ppl_{args.eval_type}.csv", index=False)

    if pred_results_dict:
        pred_results_df = pd.DataFrame.from_dict(pred_results_dict)
        pred_results_df.index = [args.decoding_type] if args.decoding_type != "all" else SUPPORTED_DECODING_TYPES
        print(pred_results_df)
        print()
        pred_results_df.to_csv(f"evals_output_pred_{args.eval_type}_{args.decoding_type}.csv")