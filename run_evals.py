from tqdm import tqdm, trange
import re
import torch
import multiprocessing as mp
import random
from torch.nn import CrossEntropyLoss

from realtime_chatbot.utils import args_helpers
from realtime_chatbot.realtime_agent import RealtimeAgent, RealtimeAgent_Resources, RealtimeAgentConfig

def get_agent(args, device=None):
    agent = RealtimeAgent(
        resources=RealtimeAgent_Resources(modelpath=args.agent_modelpath, device=device),
        config=RealtimeAgentConfig(
            random_state=args.random_state,
            prevent_special_token_generation=args.prevent_special_token_generation,
            add_special_pause_token=args.add_special_pause_token)
    )
    return agent

def load_test_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.readlines()

def get_trp_examples(test_data):
    speakers_in_prefix_regex = re.compile(r"S\d+(?= \(name:)")
    speakers_in_transcript_regex = re.compile(r"S\d+(?=:)")
    positive_trp_examples = []
    negative_trp_examples = []
    transcript_marker = "Transcript: "
    for example in tqdm(test_data, desc="Preparing TRP Examples"):
        prefix, transcript = example.split(transcript_marker)
        prefix += transcript_marker
        speakers = re.findall(speakers_in_prefix_regex, prefix)
        speakers_in_transcript = set(re.findall(speakers_in_transcript_regex, transcript))
        # If there are less than 2 active speakers in the transcript, no turn-taking takes place and it doesn't
        # make sense to compute a TRP loss for this example
        if len(speakers_in_transcript) < 2:
            continue
        transcript_words = transcript.split()
        current_speaker = None
        for i, word in enumerate(transcript_words):
            word_speaker_identity_match = re.match(speakers_in_transcript_regex, word)
            if current_speaker is not None:
                transcript_history = f"{prefix}{' '.join(transcript_words[:i])}"
                # If we are at a turn switch to another speaker, compute the positive TRP loss (lower is better).
                # If we are not at a turn switch, compute the negative TRP loss (higher is better).
                if word_speaker_identity_match:
                    # Positive TRP only makes sense for a turn-switch to another speaker.
                    # If we are at a turn switch to the same speaker (repeated speaker identity for a new utterance), do nothing.
                    if word_speaker_identity_match[0] != current_speaker:
                        positive_trp_examples.append(f"{transcript_history} {word_speaker_identity_match[0]}:")
                else:
                    for speaker in speakers:
                        if speaker != current_speaker:
                            negative_trp_examples.append(f"{transcript_history} {speaker}:")
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

def eval_worker_init(queue, args):
    device_id = queue.get()
    device = torch.device(f"cuda:{device_id}")
    global worker_agent
    worker_agent = get_agent(args, device=device)
    global worker_args
    worker_args = args

def eval_with_worker(batch):
    return get_trp_losses(worker_agent, batch, worker_args.batch_size, show_progress=False)

def eval_trp_ppl(test_data, args):
    positive_trp_examples, negative_trp_examples = get_trp_examples(test_data)

    n_workers = torch.cuda.device_count()
    if n_workers > 1:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for i in range(n_workers):
            queue.put(i)
        positive_trp_losses = []
        negative_trp_losses = []
        with ctx.Pool(n_workers, initializer=eval_worker_init, initargs=(queue, args)) as pool:
            for examples, losses in ((positive_trp_examples, positive_trp_losses), (negative_trp_examples, negative_trp_losses)):
                batches = [examples[start:start+args.batch_size] for start in range(0, len(examples), args.batch_size)]
                with tqdm(total=len(batches), desc="Computing TRP Losses") as pbar:
                    for res in pool.imap_unordered(eval_with_worker, batches):
                        losses.extend(res)
                        pbar.update()
    else:
        agent = get_agent(args)
        positive_trp_losses = get_trp_losses(agent, positive_trp_examples, args.batch_size)
        negative_trp_losses = get_trp_losses(agent, negative_trp_examples, args.batch_size)

    positive_trp_ppl = torch.exp(torch.tensor(positive_trp_losses).mean()).item()
    negative_trp_ppl = torch.exp(torch.tensor(negative_trp_losses).mean()).item()
    return positive_trp_ppl, negative_trp_ppl

if __name__ == "__main__":
    parser = args_helpers.get_common_arg_parser()
    parser.add_argument("--test-data", default="data/dataset_test.txt")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--data-random-state", type=int, default=42)
    args = parser.parse_args()

    print("\nRunning with arguments:")
    print(args)
    print()

    test_data = load_test_data(args.test_data)
    if args.num_examples > 0:
        random.seed(args.data_random_state)
        test_data = random.sample(test_data, args.num_examples)

    positive_trp_ppl, negative_trp_ppl = eval_trp_ppl(test_data, args)
    print(f"Positive TRP PPL: {positive_trp_ppl}")
    print(f"Negative TRP PPL: {negative_trp_ppl}")




