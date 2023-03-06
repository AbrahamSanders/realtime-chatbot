import argparse
from os import path

from realtime_chatbot.data_loaders.talkbank_data_loader import TalkbankDataLoader
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser("Prepare the training dataset")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--corpora", default="All")
    parser.add_argument("--summarization-modelname", default="lidiya/bart-large-xsum-samsum")
    parser.add_argument("--test-proportion", type=float, default=0.1)
    parser.add_argument("--dev-proportion", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    if args.summarization_modelname == "None":
        args.summarization_modelname = None
    
    loader = TalkbankDataLoader(summarization_modelname=args.summarization_modelname)
    examples = list(loader.load_data(corpora=args.corpora, exclude="MICASE.+?(?:lab500su044|ofc301mu021)"))
    
    train_examples, test_examples = train_test_split(examples, test_size=args.test_proportion, random_state=args.seed)
    train_examples, dev_examples = train_test_split(train_examples, test_size=args.dev_proportion, random_state=args.seed)
    
    for split, split_examples in zip(("train", "dev", "test"), (train_examples, dev_examples, test_examples)):
        output_filename = path.join(args.data_dir, f"dataset_{split}.txt")
        with open(output_filename, "w", encoding="utf-8") as f:
            for example in split_examples:
                f.write(example)
                f.write("\n")
    
if __name__ == "__main__":
    main()
    