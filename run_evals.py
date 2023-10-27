import random
import pandas as pd
from datetime import datetime
from scipy.stats import hmean

from realtime_chatbot.utils import args_helpers
from realtime_chatbot.evals import common as cm
from realtime_chatbot.evals.turn_taking import eval_trp_ppl
from realtime_chatbot.evals.pausing import eval_pause_ppl
from realtime_chatbot.evals.response_quality import eval_response_quality

def get_filename_suffix(option_list):
    if "all" in option_list:
        return "all"
    elif len(option_list) == 1:
        return option_list[0]
    else:
        return "multiple"

if __name__ == "__main__":
    parser = args_helpers.get_common_arg_parser()
    parser.add_argument("--test-data", default="data/dataset_test.txt")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--contrastive-batch-size", type=int, default=2)
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--data-random-state", type=int, default=42)
    parser.add_argument("--eval-type", choices=["all"] + cm.SUPPORTED_EVAL_TYPES, default=["all"], nargs="+")
    parser.add_argument("--decoding-type", choices=["all"] + cm.SUPPORTED_DECODING_TYPES, default=["all"], nargs="+")
    parser.add_argument("--use-fp16" , action="store_true", default=False)
    parser.add_argument("--use-bf16" , action="store_true", default=False)
    args = parser.parse_args()

    if args.random_state is None:
        print("\nrandom_state not set. Setting to 42...")
        args.random_state = 42

    print("\nRunning with arguments:")
    print(args)
    print()

    eval_type_filename_suffix = get_filename_suffix(args.eval_type)
    if "all" in args.eval_type:
        args.eval_type = cm.SUPPORTED_EVAL_TYPES

    decoding_type_filename_suffix = get_filename_suffix(args.decoding_type)
    if "all" in args.decoding_type:
        args.decoding_type = cm.SUPPORTED_DECODING_TYPES

    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    test_data = cm.load_test_data(args.test_data)
    if args.num_examples > 0:
        random.seed(args.data_random_state)
        test_data = random.sample(test_data, args.num_examples)

    ppl_results_dict = {}
    pred_results_dict = {}
    worker_pool, decoding_type = cm.setup_worker_pool(args)
    with worker_pool:
        # Turn-taking & Pausing Perplexity evals
        cm.eval_and_print("ppl_trp", eval_trp_ppl, None, worker_pool, args, test_data, ppl_results_dict)
        cm.eval_and_print("ppl_pause", eval_pause_ppl, None, worker_pool, args, test_data, ppl_results_dict)

        # Prec, Rec, F1 evals for turn taking, pausing, fillers, backchannels, and laughter
        cm.eval_and_print("pred", cm.eval_pred, decoding_type, worker_pool, args, test_data, pred_results_dict)

        # Response quality evals
        cm.eval_and_print("response", eval_response_quality, decoding_type, worker_pool, args, test_data, pred_results_dict)

    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    runtime_minutes = int((end_time - start_time).total_seconds() / 60)
    print(f"Total time: {runtime_minutes} minutes.")

    if ppl_results_dict:
        ppl_results_df = pd.DataFrame.from_dict(ppl_results_dict)
        print(ppl_results_df)
        print()
        ppl_results_df.to_csv(f"evals_output_ppl_{eval_type_filename_suffix}.csv", index=False)

    if pred_results_dict:
        pred_results_df = pd.DataFrame.from_dict(pred_results_dict)
        pred_results_df.index = args.decoding_type

        # overall metrics, computed from all except precision (prec) and recall (rec) 
        # because they are redundant with f1 when averaged
        non_verbal_metrics_to_include = [metric for metric in pred_results_dict if metric.startswith("pred_") and metric.endswith("_f1")]
        verbal_metrics_to_include = [metric for metric in pred_results_dict if metric.startswith("response_")]
        pred_results_df["avg_non_verbal"] = pred_results_df[non_verbal_metrics_to_include].mean(axis=1)
        pred_results_df["avg_verbal"] = pred_results_df[verbal_metrics_to_include].mean(axis=1)
        # overall is the harmonic mean between avg_non_verbal and avg_verbal
        pred_results_df["overall"] = hmean(pred_results_df[["avg_non_verbal", "avg_verbal"]], axis=1)

        print(pred_results_df)
        print()
        suffix = f"{eval_type_filename_suffix}_{decoding_type_filename_suffix}"
        pred_results_df.to_csv(f"evals_output_pred_{suffix}.csv")