from tqdm import tqdm
from scipy.stats import ttest_ind
import re

from . import common as cm
from .data_processing import speaker_regex, pause_regex
from .metrics_simctg import measure_repetition_and_diversity
from .metrics import measure_generic_responses

def get_response_examples(test_data):
    response_examples = []
    transcript_marker = "Transcript: "
    for example in tqdm(test_data, desc="Preparing Response Examples"):
        prefix, transcript = example.split(transcript_marker)
        prefix += transcript_marker
        transcript_words = transcript.split()
        for i, word in enumerate(transcript_words):
            if re.match(speaker_regex, word):
                transcript_history = f"{prefix}{' '.join(transcript_words[:i+1])}"
                response_examples.append(transcript_history)

    # sort examples by length for greater batch efficiency
    response_examples.sort(key=len, reverse=True)
    return response_examples

def get_responses_with_worker(batch):
    eval_decoding_type = cm.worker_decoding_type.value.decode("utf-8")
    batch_size = cm.get_batch_size(cm.worker_args, eval_decoding_type)
    return cm.get_predictions(cm.worker_agent, batch, batch_size, eos_token=" S", strip_eos_token=True, 
                              max_new_tokens=60, decoding_type=eval_decoding_type, show_progress=False)

def eval_response_quality(worker_pool, args, test_data, batch_size):
    examples = get_response_examples(test_data)
    print(f"# Response Prediction examples: {len(examples)}")
    print()

    predictions = cm.get_model_output_with_worker_pool(worker_pool, examples, batch_size, get_responses_with_worker)

    # get distribution of pauses from test data
    gt_pauses = []
    for test_example in test_data:
        gt_pauses.extend(re.findall(pause_regex, test_example))
    gt_pauses = [float(pause) for pause in gt_pauses]

    # get distribution of pauses from predictions
    pred_pauses = []
    for prediction in predictions:
        pred_pauses.extend(re.findall(pause_regex, prediction))
    pred_pauses = [float(pause) for pause in pred_pauses]

    # t-test
    _, pause_pvalue = ttest_ind(gt_pauses, pred_pauses, equal_var=False)

    # we don't consider pauses in diversity calculation
    predictions = [re.sub(pause_regex, "", response) for response in predictions]
    predictions = [re.sub(" {2,}", " ", response) for response in predictions]
    predictions = [response.strip() for response in predictions]
    predictions = [response for response in predictions if response != ""]

    _, _, _, diversity = measure_repetition_and_diversity(predictions)
    generic = measure_generic_responses(predictions)

    return [
        ("pause_dist_similarity", 1.0-pause_pvalue),
        ("diversity", diversity),
        ("specific", 1.0-generic)
    ]