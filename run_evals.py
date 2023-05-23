import random
import pandas as pd
from datetime import datetime

from realtime_chatbot.utils import args_helpers
from realtime_chatbot.evals import common as cm
from realtime_chatbot.evals.turn_taking import eval_trp_ppl, eval_trp_pred
from realtime_chatbot.evals.pausing import eval_pause_ppl, eval_pause_pred
from realtime_chatbot.evals.response import eval_response_pred

if __name__ == "__main__":
    parser = args_helpers.get_common_arg_parser()
    parser.add_argument("--test-data", default="data/dataset_test_dyads_original_pauses.txt")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--contrastive-batch-size", type=int, default=5)
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--data-random-state", type=int, default=42)
    parser.add_argument("--eval-type", choices=["all", "trp_ppl", "trp_pred", "pause_ppl", "pause_pred", "response_pred"], default="all")
    parser.add_argument("--decoding-type", choices=["all"] + cm.SUPPORTED_DECODING_TYPES, default="all")
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

    test_data = cm.load_test_data(args.test_data)
    if args.num_examples > 0:
        random.seed(args.data_random_state)
        test_data = random.sample(test_data, args.num_examples)

    ppl_results_dict = {}
    pred_results_dict = {}
    worker_pool, decoding_type = cm.setup_worker_pool(args)
    with worker_pool:
        # Turn-taking Evals
        cm.eval_and_print("trp_ppl", eval_trp_ppl, None, worker_pool, args, test_data, ppl_results_dict)
        cm.eval_and_print("trp_pred", eval_trp_pred, decoding_type, worker_pool, args, test_data, pred_results_dict)

        # Pausing Evals
        cm.eval_and_print("pause_ppl", eval_pause_ppl, None, worker_pool, args, test_data, ppl_results_dict)
        cm.eval_and_print("pause_pred", eval_pause_pred, decoding_type, worker_pool, args, test_data, pred_results_dict)

        # Response Evals
        cm.eval_and_print("response_pred", eval_response_pred, decoding_type, worker_pool, args, test_data, pred_results_dict)

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
        pred_results_df.index = [args.decoding_type] if args.decoding_type != "all" else cm.SUPPORTED_DECODING_TYPES
        print(pred_results_df)
        print()
        pred_results_df.to_csv(f"evals_output_pred_{args.eval_type}_{args.decoding_type}.csv")