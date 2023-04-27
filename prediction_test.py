from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed
import torch
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import re

from realtime_chatbot.utils import args_helpers
from realtime_chatbot.dynamic_contrastive import get_contrastive_search_override

def get_process_method(model, tokenizer, device):

    def predict_completion(sequence, similarity_threshold, seed, max_new_tokens, 
                           last_prediction_context_pos, last_prediction, last_prediction_embed, 
                           last_prediction_response, **generate_kwargs):
        similarity_with_last_pred = 0.0
        input_ids = None
        if last_prediction_context_pos is not None:
            last_pred_context, actual_completion = sequence[:last_prediction_context_pos], sequence[last_prediction_context_pos:]
            last_pred_context_tokens = tokenizer.encode(last_pred_context, return_tensors="pt").to(device)
            actual_completion_tokens = tokenizer.encode(actual_completion, return_tensors="pt", add_special_tokens=False).to(device)
            input_ids = torch.cat((last_pred_context_tokens, actual_completion_tokens), dim=1)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
                actual_completion_embed = outputs.hidden_states[-1][:, last_pred_context_tokens.shape[-1]:].mean(dim=1)
                similarity_with_last_pred = torch.cosine_similarity(last_prediction_embed, actual_completion_embed, dim=-1).item()

        if seed:
            set_seed(int(seed))

        if similarity_with_last_pred >= similarity_threshold:
            # If last prediction is similar enough to the actual completion, no need to make a new prediction. Just check if we're at a turn-switch.
            prediction_context_pos, prediction, prediction_embed, prediction_response = (
                last_prediction_context_pos, last_prediction, last_prediction_embed, last_prediction_response
            )
            outputs = model.generate(input_ids=input_ids, max_new_tokens=1, **generate_kwargs)
            is_turn_switch = outputs[0, -1].item() == generate_kwargs["eos_token_id"]
        else:
            # Otherwise, make a new utterance completion prediction.
            if input_ids is None:
                input_ids = tokenizer.encode(sequence, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids=input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=max_new_tokens, **generate_kwargs
            )
            prediction_context_pos = len(sequence)
            prediction = tokenizer.decode(outputs.sequences[0, input_ids.shape[-1]:-1], skip_special_tokens=False)
            prediction_embed = None
            is_turn_switch = True
            if prediction:
                prediction_embed = torch.cat([pos_states[-1] for pos_states in outputs.hidden_states[1:]], dim=1).mean(dim=1)
                is_turn_switch = False
            # Generate a response to the predicted completion.
            input_ids = outputs.sequences
            outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **generate_kwargs)
            prediction_response = tokenizer.decode(outputs[0, input_ids.shape[-1]-1:-1], skip_special_tokens=False)

        return similarity_with_last_pred, prediction_context_pos, prediction, prediction_embed, prediction_response, is_turn_switch

    def process_text(input_text, step_size, similarity_threshold, seed, decoding, max_new_tokens, temperature, num_beams, 
                        top_k, top_p, typical_p, min_penalty_alpha, max_penalty_alpha):
        
        generate_kwargs = {
                "num_beams": int(num_beams),
                "early_stopping": True,
                "eos_token_id": tokenizer(" S", add_special_tokens=False).input_ids[0]
        }
        if decoding != "Greedy":
            generate_kwargs["top_k"] = int(top_k)
        if decoding == "Sampling":
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = float(temperature)
            generate_kwargs["top_p"] = float(top_p)
            generate_kwargs["typical_p"] = float(typical_p)
        if decoding == "Contrastive":
            generate_kwargs["penalty_alpha"] = float(max_penalty_alpha)
            model.contrastive_search = get_contrastive_search_override(model, float(min_penalty_alpha), float(max_penalty_alpha), 
                                                                       sample_top_p=float(top_p), sample_temperature=float(temperature))

        results_dict = {
            "Context": [],
            "Actual Completion": [],
            "Sim. With Last Pred": [],
            "Predicted Completion": [],
            "Predicted Response": []
        }
        context, actual_completion = input_text.split("|")
        actual_completion_words = re.findall(" *[^ ]+", actual_completion)
        
        similarity_with_last_pred = 0.0
        last_prediction_context_pos = None
        last_prediction = None
        last_prediction_embed = None
        last_prediction_response = None
        while True:
            similarity_with_last_pred, prediction_context_pos, prediction, prediction_embed, prediction_response, is_turn_switch = predict_completion(
                context, 
                float(similarity_threshold), 
                seed, 
                int(max_new_tokens), 
                last_prediction_context_pos, 
                last_prediction, 
                last_prediction_embed, 
                last_prediction_response, 
                **generate_kwargs
            )
            display_context_pos = last_prediction_context_pos if last_prediction_context_pos is not None else prediction_context_pos
            results_dict["Context"].append(f"...{context[display_context_pos-30:display_context_pos]}")
            results_dict["Actual Completion"].append(context[display_context_pos:])
            results_dict["Sim. With Last Pred"].append(similarity_with_last_pred)
            results_dict["Predicted Completion"].append(prediction if prediction != last_prediction else "^^^^^^")
            results_dict["Predicted Response"].append(prediction_response if prediction_response != last_prediction_response else "^^^^^^")

            if is_turn_switch or len(actual_completion_words) == 0:
                break
            last_prediction_context_pos = prediction_context_pos
            last_prediction = prediction
            last_prediction_embed = prediction_embed
            last_prediction_response = prediction_response

            # prepare context for next step
            context += "".join(actual_completion_words[:step_size])
            actual_completion_words = actual_completion_words[step_size:]

        results_df = pd.DataFrame.from_dict(results_dict)
        similarities_plot = go.Figure(data=[
            go.Scatter(x=list(range(1, results_df.shape[0]+1)), y=results_dict["Sim. With Last Pred"]),
        ]).update_layout({
            "yaxis": {
                "range": [0.0, 1.0]
            }
        })
        return results_df, similarities_plot

    return process_text

if __name__ == "__main__":
    parser = args_helpers.get_common_arg_parser()
    args = parser.parse_args()

    print("\nRunning with arguments:")
    print(args)
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.agent_modelpath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.agent_modelpath, torch_dtype=torch.float16).to(device)

    interface = gr.Interface(
        fn=get_process_method(model, tokenizer, device),
        inputs=[
            gr.Textbox(label="Input", lines=4),
            gr.Slider(1, 10, value=5, step=1, label="Step Size (simulated spoken input speed)"),
            gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Similarity Threshold"),
            gr.Textbox(label="Random seed", value="42"),
            gr.Radio(["Greedy", "Sampling", "Contrastive"], value="Contrastive", label="Decoding"),
            gr.Slider(1, 300, value=30, step=1, label="Max New Tokens"),
            gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Temperature"),
            gr.Slider(1, 10, value=1, step=1, label="Beams"),
            gr.Slider(0, 100, value=8, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=0.8, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Typical-p"),
            gr.Slider(0.0, 1.0, value=0.005, step=0.005, label="Min Penalty-alpha"),
            gr.Slider(0.0, 1.0, value=1.0, step=0.005, label="Max Penalty-alpha")
        ], 
        outputs=[
            gr.DataFrame(label="Results", wrap=True),
            gr.Plot(label="Similarity")
        ],
        allow_flagging='never'
    )
    interface.launch()