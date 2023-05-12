from tqdm import tqdm
import re
import torch
from torch.nn import CrossEntropyLoss

from realtime_chatbot.utils import args_helpers
from realtime_chatbot.realtime_agent import RealtimeAgent, RealtimeAgent_Resources, RealtimeAgentConfig

def load_test_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.readlines()

def get_token_losses(agent, text):
    inputs = agent.resources.tokenizer(text, return_tensors="pt").to(agent.resources.device)
    with torch.no_grad():
        logits = agent.resources.model(**inputs).logits
    labels = inputs.input_ids
    # From https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/opt/modeling_opt.py#L953
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(shift_logits.view(-1, agent.resources.model.config.vocab_size), shift_labels.view(-1))
    return token_losses

def eval_trp_ppl(agent, test_data):
    positive_trp_losses = []
    negative_trp_losses = []
    transcript_marker = "Transcript: "
    for example in tqdm(test_data, desc="Evaluating TRP PPL..."):
        prefix, transcript = example.split(transcript_marker)
        prefix += transcript_marker
        transcript_words = transcript.split()
        current_speaker = None
        for i, word in enumerate(transcript_words):
            if current_speaker is not None:
                transcript_history = f"{prefix}{' '.join(transcript_words[:i])}"
                proposed_speaker = agent.config.agent_identity if current_speaker == agent.config.user_identity \
                     else agent.config.user_identity
                transcript_history += f" {proposed_speaker}:"
                token_losses = get_token_losses(agent, transcript_history)
                trp_loss = token_losses[-3:].mean()
                if transcript_history.endswith(word):
                    positive_trp_losses.append(trp_loss)
                else:
                    negative_trp_losses.append(trp_loss)

            if re.match(agent.any_identity_regex, word):
                current_speaker = word.rstrip(":")

    positive_trp_ppl = torch.exp(torch.stack(positive_trp_losses).mean()).item()
    negative_trp_ppl = torch.exp(torch.stack(negative_trp_losses).mean()).item()
    return positive_trp_ppl, negative_trp_ppl

if __name__ == "__main__":
    parser = args_helpers.get_common_arg_parser()
    parser.add_argument("--test-data", default="data/dataset_test.txt")
    args = parser.parse_args()

    print("\nRunning with arguments:")
    print(args)
    print()

    agent = RealtimeAgent(
        resources=RealtimeAgent_Resources(modelpath=args.agent_modelpath),
        config=RealtimeAgentConfig(
            random_state=args.random_state,
            prevent_special_token_generation=args.prevent_special_token_generation,
            add_special_pause_token=args.add_special_pause_token)
    )

    test_data = load_test_data(args.test_data)
    test_data = test_data[:3]

    positive_trp_ppl, negative_trp_ppl = eval_trp_ppl(agent, test_data)
    print(f"Positive TRP PPL: {positive_trp_ppl}")
    print(f"Negative TRP PPL: {negative_trp_ppl}")




