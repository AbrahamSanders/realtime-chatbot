from tqdm import tqdm, trange
import torch
import multiprocessing as mp
import ctypes
import pandas as pd
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score

from ..realtime_agent import RealtimeAgent, RealtimeAgent_Resources, RealtimeAgentConfig
from ..dynamic_contrastive import get_contrastive_search_override

from .data_processing import get_prediction_examples, classes

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

def get_model_output_with_worker_pool(worker_pool, examples, batch_size, batch_get_output_fn):
    worker_outputs = []
    batches = [examples[start:start+batch_size] for start in range(0, len(examples), batch_size)]
    with tqdm(total=len(examples), desc="Computing Model Output") as pbar:
        for res in worker_pool.imap(batch_get_output_fn, batches):
            worker_outputs.extend(res)
            pbar.update(len(res))
    return worker_outputs

def get_preds_with_worker(batch):
    eval_decoding_type = worker_decoding_type.value.decode("utf-8")
    batch_size = get_batch_size(worker_args, eval_decoding_type)
    return get_predictions(worker_agent, batch, batch_size, eos_token=worker_agent.resources.tokenizer.eos_token,
                           strip_eos_token=True, max_new_tokens=20, decoding_type=eval_decoding_type, show_progress=False)

def eval_pred(worker_pool, args, test_data, batch_size):
    examples, labels_df = get_prediction_examples(test_data)
    print(f"# Examples: {len(examples)}")
    print()

    predictions = get_model_output_with_worker_pool(worker_pool, examples, batch_size, get_preds_with_worker)

    pred_labels = []
    for example, pred in zip(examples, predictions):
        pred = pred.strip()
        pred_labels.append(tuple([criterion(example, pred) for criterion in classes.values()]))
    pred_labels_df = pd.DataFrame(pred_labels, columns=classes.keys())
    pred_labels_df[pred_labels_df == -1] = 0

    metrics = []
    for class_name in classes.keys():
        class_targets = labels_df[class_name]
        class_preds = pred_labels_df[class_name]

        include = class_targets != -1
        class_targets = class_targets[include]
        class_preds = class_preds[include]

        prec = precision_score(class_targets, class_preds)
        recall = recall_score(class_targets, class_preds)
        f1 = f1_score(class_targets, class_preds)
        metrics.append((f"{class_name}_prec", prec))
        metrics.append((f"{class_name}_rec", recall))
        metrics.append((f"{class_name}_f1", f1))

    return metrics

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