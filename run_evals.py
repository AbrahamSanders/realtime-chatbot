from tqdm import tqdm, trange
import re
import torch
import multiprocessing as mp
import random
import ctypes
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers.trainer_utils import set_seed

from realtime_chatbot.utils import args_helpers
from realtime_chatbot.realtime_agent import RealtimeAgent, RealtimeAgent_Resources, RealtimeAgentConfig
from realtime_chatbot.dynamic_contrastive import get_contrastive_search_override

speakers_in_prefix_regex = re.compile(r"S\d+(?= \(name:)")
speakers_in_transcript_regex = re.compile(r"S\d+(?=:)")

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
        agent.generate_kwargs["top_p"] = 0.8
    if "contrastive" in decoding_type:
        agent.generate_kwargs["penalty_alpha"] = 0.5
        agent.generate_kwargs["top_k"] = 8
    if decoding_type == "dynamic_contrastive":
        agent.resources.model.contrastive_search = get_contrastive_search_override(agent.resources.model, 0.005, 0.6)
    if decoding_type == "dynamic_contrastive_sampling":
        # do_sample is not actually recognized by Generate for contrastive search. Setting it here to signal
        # to the evaluation routine that we want to generate multiple samples (the override supports sampling
        # but is not "officially" a sampling decoding type).
        agent.generate_kwargs["do_sample"] = True
        agent.resources.model.contrastive_search = get_contrastive_search_override(agent.resources.model, 0.005, 1.0, sample_top_p=0.8)
        
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

    #sort examples by length for greater batch efficiency
    positive_trp_examples.sort(key=len, reverse=True)
    negative_trp_examples.sort(key=len, reverse=True)
    return positive_trp_examples, negative_trp_examples

def get_trp_losses(agent, examples, batch_size, show_progress=True):
    trp_losses = []
    turn_switch_token_id = agent.resources.tokenizer(" S", add_special_tokens=False).input_ids[0]

    for start_index in trange(0, len(examples), batch_size, desc="Computing TRP Losses", disable=not show_progress):
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
            last_speaker_idx = (shift_labels[i] == turn_switch_token_id).nonzero(as_tuple=True)[0][-1]
            trp_loss = token_losses[i, last_speaker_idx:actual_length-1].mean()
            trp_losses.append(trp_loss.item())

    return trp_losses

def get_trp_predictions(agent, examples, decoding_type, random_state, show_progress=True):
    trp_at_1_preds = []
    trp_at_5_preds = []

    set_generate_kwargs(agent, decoding_type)

    do_sample = agent.generate_kwargs.get("do_sample", False)
    num_return_sequences = 5 if do_sample else 1
    num_generate_calls = 1
    if do_sample and "penalty_alpha" in agent.generate_kwargs:
        # special case for dynamic contrastive search. Generate does not support do_sample=True or num_return_sequences > 1
        # for contrastive search, so we need to do multiple generate calls and pass do_sample=False 
        # (the override does the sampling regardless if sample_top_p is set)
        do_sample = False
        num_generate_calls = num_return_sequences
        num_return_sequences = 1

    colon_token_id = agent.resources.tokenizer(":", add_special_tokens=False).input_ids[0]
    for example in tqdm(examples, desc="Computing TRP Predictions", disable=not show_progress):
        current_speaker = re.findall(speakers_in_transcript_regex, example)[-1]
        if random_state is not None:
            set_seed(random_state)
        predictions = []
        for _ in range(num_generate_calls):
            generated_preds = agent._generate(example, do_sample=do_sample, num_return_sequences=num_return_sequences, 
                                              eos_token_id=colon_token_id, max_new_tokens=4)
            predictions.extend(generated_preds if num_return_sequences > 1 else [generated_preds])

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
        if num_return_sequences == 5:
            trp_at_5_preds.append(trp_at_5)
        
    return trp_at_1_preds, trp_at_5_preds

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
    return get_trp_losses(worker_agent, batch, worker_args.batch_size, show_progress=False)

def eval_trp_preds_with_worker(batch):
    eval_decoding_type = worker_decoding_type.value.decode("utf-8")
    return get_trp_predictions(worker_agent, batch, eval_decoding_type, worker_args.random_state, show_progress=False)

def eval_trp_ppl(worker_pool, test_data, batch_size):
    positive_trp_examples, negative_trp_examples = get_trp_examples(test_data, "ppl")

    positive_trp_losses = []
    negative_trp_losses = []
    for examples, losses in ((positive_trp_examples, positive_trp_losses), (negative_trp_examples, negative_trp_losses)):
        batches = [examples[start:start+batch_size] for start in range(0, len(examples), batch_size)]
        with tqdm(total=len(batches), desc="Computing TRP Losses") as pbar:
            for res in worker_pool.imap_unordered(eval_trp_losses_with_worker, batches):
                losses.extend(res)
                pbar.update()

    positive_trp_ppl = torch.exp(torch.tensor(positive_trp_losses).mean()).item()
    negative_trp_ppl = torch.exp(torch.tensor(negative_trp_losses).mean()).item()
    return positive_trp_ppl, negative_trp_ppl

def eval_trp_prediction(worker_pool, test_data):
    positive_trp_examples, negative_trp_examples = get_trp_examples(test_data, "pred")
    trp_examples = positive_trp_examples + negative_trp_examples
    trp_targets = [1] * len(positive_trp_examples) + [0] * len(negative_trp_examples)

    trp_at_1_preds = []
    trp_at_5_preds = []
    batches = [[example] for example in trp_examples]
    with tqdm(total=len(batches), desc="Computing TRP Predictions") as pbar:
        for res in worker_pool.imap(eval_trp_preds_with_worker, batches):
            trp_at_1_preds.extend(res[0])
            trp_at_5_preds.extend(res[1])
            pbar.update()

    prec_at_1 = precision_score(trp_targets, trp_at_1_preds)
    recall_at_1 = recall_score(trp_targets, trp_at_1_preds)
    f1_at_1 = f1_score(trp_targets, trp_at_1_preds)
    
    prec_at_5 = None
    recall_at_5 = None
    f1_at_5 = None
    if len(trp_at_5_preds) > 0:
        prec_at_5 = precision_score(trp_targets, trp_at_5_preds)
        recall_at_5 = recall_score(trp_targets, trp_at_5_preds)
        f1_at_5 = f1_score(trp_targets, trp_at_5_preds)

    return prec_at_1, recall_at_1, f1_at_1, prec_at_5, recall_at_5, f1_at_5

if __name__ == "__main__":
    parser = args_helpers.get_common_arg_parser()
    parser.add_argument("--test-data", default="data/dataset_test.txt")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--data-random-state", type=int, default=42)
    parser.add_argument("--eval-type", choices=["all", "trp_ppl", "trp_pred"], default="all")
    parser.add_argument("--decoding-type", choices=[
        "all", "greedy", "nucleus", "contrastive", "dynamic_contrastive", "dynamic_contrastive_sampling"], default="all")
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

    test_data = load_test_data(args.test_data)
    if args.num_examples > 0:
        random.seed(args.data_random_state)
        test_data = random.sample(test_data, args.num_examples)

    worker_pool, decoding_type = setup_worker_pool(args)
    with worker_pool:
        if args.eval_type == "all" or args.eval_type == "trp_ppl":
            print("-------------------------------------------------")
            print("-- Evaluating TRP PPL...")
            print("-------------------------------------------------")
            print()
            positive_trp_ppl, negative_trp_ppl = eval_trp_ppl(worker_pool, test_data, args.batch_size)
            print(f"Positive TRP PPL: {positive_trp_ppl}")
            print(f"Negative TRP PPL: {negative_trp_ppl}")
            print()

        if args.eval_type == "all" or args.eval_type == "trp_pred":
            eval_decoding_types = [args.decoding_type] if args.decoding_type != "all" else [
                "greedy", "nucleus", "contrastive", "dynamic_contrastive", "dynamic_contrastive_sampling"
            ]
            for eval_decoding_type in eval_decoding_types:
                decoding_type.value = eval_decoding_type.encode("utf-8")
                print("-------------------------------------------------")
                print("-- Evaluating TRP Prediction...")
                print(f"-- (decoding: {eval_decoding_type})")
                print("-------------------------------------------------")
                print()
                prec_at_1, recall_at_1, f1_at_1, prec_at_5, recall_at_5, f1_at_5 = eval_trp_prediction(worker_pool, test_data)
                print(f"Precision@1: {prec_at_1}")
                print(f"Recall@1: {recall_at_1}")
                print(f"F1@1: {f1_at_1}")
                print()
                if prec_at_5 is not None:
                    print(f"Precision@5: {prec_at_5}")
                    print(f"Recall@5: {recall_at_5}")
                    print(f"F1@5: {f1_at_5}")