import argparse
from os import path
from itertools import chain

from realtime_chatbot.data_loaders.talkbank_data_loader import TalkbankDataLoader
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser("Prepare the training dataset")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--corpora", default="All")
    parser.add_argument("--standardize-pauses", action="store_true")
    parser.add_argument("--add-special-pause-token", action="store_true")
    parser.add_argument("--summarization-modelname", default="lidiya/bart-large-xsum-samsum")
    parser.add_argument("--test-proportion", type=float, default=0.1)
    parser.add_argument("--dev-proportion", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    if args.summarization_modelname == "None":
        args.summarization_modelname = None
    
    loader = TalkbankDataLoader(
        standardize_pauses=args.standardize_pauses, 
        add_special_pause_token=args.add_special_pause_token,
        summarization_modelname=args.summarization_modelname, 
        random_state=args.seed
    )
    dialogues = list(loader.load_data(corpora=args.corpora, exclude="MICASE.+?(?:lab500su044|ofc301mu021)", group_by_dialogue=True))
    
    train_dialogues, test_dialogues = train_test_split(dialogues, test_size=args.test_proportion, random_state=args.seed)
    train_dialogues, dev_dialogues = train_test_split(train_dialogues, test_size=args.dev_proportion, random_state=args.seed)
    train_examples = list(chain(*train_dialogues))
    dev_examples = list(chain(*dev_dialogues))
    test_examples = list(chain(*test_dialogues))

    for split, split_examples in zip(("train", "dev", "test"), (train_examples, dev_examples, test_examples)):
        output_filename = path.join(args.data_dir, f"dataset_{split}.txt")
        with open(output_filename, "w", encoding="utf-8") as f:
            for example in split_examples:
                f.write(example)
                f.write("\n")
    
if __name__ == "__main__":
    main()
    