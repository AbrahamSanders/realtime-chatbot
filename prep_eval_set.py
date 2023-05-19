import argparse
import re
from os import path

from run_evals import speakers_in_prefix_regex
pauses_in_transcript_regex = re.compile(r"(?:<p> )?\((\d*?\.{1,2}\d*?)\)")

def main():
    parser = argparse.ArgumentParser("Prepare the test set for evals")
    parser.add_argument("--data-dir", default="data")
    
    args = parser.parse_args()

    raw_test_examples = None
    raw_test_set_path = path.join(args.data_dir, "raw", "dataset_test.txt")
    if path.exists(raw_test_set_path):
        with open(raw_test_set_path, encoding="utf-8") as f:
            raw_test_examples = f.readlines()

    test_set_path = path.join(args.data_dir, "dataset_test.txt")
    with open(test_set_path, encoding="utf-8") as f:
        test_examples = f.readlines()

    output_examples = []
    output_examples_raw = []
    for i, example in enumerate(test_examples):
        # is the example a dyad?
        transcript_marker = "Transcript: "
        prefix, transcript = example.split(transcript_marker)
        prefix += transcript_marker
        speakers = re.findall(speakers_in_prefix_regex, prefix)
        if len(speakers) > 2:
            continue
        
        # does the example have original timed pause annotations?
        if raw_test_examples is not None:
            raw_example = raw_test_examples[i]
            _, transcript = raw_example.split(transcript_marker)
            pauses = re.findall(pauses_in_transcript_regex, transcript)
            if len(pauses) == 0:
                continue
            num_timed_pauses = 0
            for pause in pauses:
                try:
                    _ = float(pause)
                    num_timed_pauses += 1
                except ValueError:
                    continue
            if num_timed_pauses / len(pauses) < 0.8:
                continue

        output_examples.append(example)
        if raw_test_examples is not None:
            output_examples_raw.append(raw_example)

    output_path = path.join(args.data_dir, "dataset_test_dyads_original_pauses.txt")
    print(f"Output {len(output_examples)} ({100*(len(output_examples)/len(test_examples)):.2f}% of {len(test_examples)} total) examples to {output_path}")

    with open(output_path, "w", encoding="utf-8") as fw:
        fw.writelines(output_examples)

    if len(output_examples_raw) > 0:
        with open(output_path.replace(".txt", "_raw.txt"), "w", encoding="utf-8") as fw:
            fw.writelines(output_examples_raw)
    
if __name__ == "__main__":
    main()
    