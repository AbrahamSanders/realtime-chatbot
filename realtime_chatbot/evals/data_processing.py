import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

speaker_regex = r"S\d+(?=:)"
pause_regex = r"(?:<p> )?\((\d*?\.\d*?)\)"

# 0 = not an instance of this class
# 1 = an instance of this class
# -1 = unknown (not enough information to determine - exclude from evaluation)

def is_trp(ctx, ngram):
    speaker_match = re.match(speaker_regex, ngram)
    if speaker_match is not None:
        current_speaker = re.findall(speaker_regex, ctx)[-1]
        return 1 if speaker_match[0] != current_speaker else -1
    return 0

def is_pause(ctx, ngram):
    is_pause_ = int(bool(re.match(pause_regex, ngram)))
    if is_pause_ == 0:
        # The beginning or end of a turn is a common place for speakers to pause, even if not explicitly annotated.
        # In these cases, we don't have enough information to determine whether or not this should be a pause.
        last_word = ctx[ctx.rindex(" ")+1:]
        if re.match(speaker_regex, ngram) or re.match(speaker_regex, last_word):
            is_pause_ = -1
    return is_pause_

def is_short_pause(ctx, ngram):
    is_pause_ = is_pause(ctx, ngram)
    if is_pause_ == 1:
        is_pause_ = int(float(re.match(pause_regex, ngram)[1]) < 0.7)
    return is_pause_

def is_long_pause(ctx, ngram):
    is_pause_ = is_pause(ctx, ngram)
    if is_pause_ == 1:
        is_pause_ = int(float(re.match(pause_regex, ngram)[1]) >= 0.7)
    return is_pause_

def is_filler(ctx, ngram):
    return int(bool(
        re.match(r"[ue]+r*[hm]+(?=[., ]+|\Z)", ngram, flags=re.IGNORECASE)
        or re.match(r"h+m+(?=[., ]+|\Z)", ngram, flags=re.IGNORECASE)
        or re.match(r"anyways?(?:,|\.{2,})", ngram, flags=re.IGNORECASE)
        or re.match(r"so(?:,|\.{2,})", ngram, flags=re.IGNORECASE)
        or re.match(r"like(?:,|\.{2,})", ngram, flags=re.IGNORECASE)
        or re.match(r"y(?:a|ou) know(?:,|\.{2,})", ngram, flags=re.IGNORECASE)
    ))

def is_backchannel(ctx, ngram):
    # backchannel should be two turn switches with three or less words in between
    # that contains an acknowledgement or (dis)agreement (e.g., S1: Mhm. S2:)

    # 1. Is the current ngram start with a turn switch (away from the last speaker)?
    is_trp_ = is_trp(ctx, ngram)
    if is_trp_ != 1:
        return 0
    # 2. Is the current ngram a short utterance (three or less words) followed by another speaker identity?
    short_utterance_match = re.match(r"(S\d+:(?: \S+){1,3}? )(S\d+:)", ngram)
    if short_utterance_match is None:
        return 0
    # 3. is the other speaker different from the first speaker?
    utterance, other_speaker = short_utterance_match[1], short_utterance_match[2]
    is_trp_ = is_trp(f"{ctx} {utterance}", other_speaker)
    if is_trp_ != 1:
        return 0
    # 4. Does the utterance contain an acknowledgement or (dis)agreement?
    return int(bool(
        re.search(r" m+[hm]+[.?!, ]+", utterance, flags=re.IGNORECASE)
        or re.search(r" (?:u+[hm]+){2,}[.?!, ]", utterance, flags=re.IGNORECASE)
        or re.search(r" ye?[ash]h*[.?!, ]+", utterance, flags=re.IGNORECASE)
        or re.search(r" y[eu]p[.?!, ]+", utterance, flags=re.IGNORECASE)
        or re.search(r" (?:no(?:pe)?|na+)[.?!, ]+", utterance, flags=re.IGNORECASE)
        or re.search(r"(?<!all) right[.?!, ]+", utterance, flags=re.IGNORECASE)
        or re.search(r" o+h+[.?!, ]+", utterance, flags=re.IGNORECASE)
        or re.search(r" ok(?:ay)?[.?!, ]+", utterance, flags=re.IGNORECASE)
        or re.search(r" a+h+a*[.?!, ]+", utterance, flags=re.IGNORECASE)
        or re.search(r" h+a+h*[.?!, ]+", utterance, flags=re.IGNORECASE)
        or re.search(r" eh+[.?!, ]+", utterance, flags=re.IGNORECASE)
    ))

def is_laughter(ctx, ngram):
    return int(bool(
        re.match(r"(?:&=|\[%?) ?(?:laugh|giggle|cackle)", ngram, flags=re.IGNORECASE)
        or re.match(r"b?a*(?:h+a+h* ?){2,}", ngram, flags=re.IGNORECASE)
        or re.match(r"t?e*(?:h+e+(?:h+ )?){2,}", ngram, flags=re.IGNORECASE)
        or re.match(r"o*(?:h+o+h* ?){2,}", ngram, flags=re.IGNORECASE)
    ))

classes = OrderedDict()
classes["trp"] = is_trp
classes["pause"] = is_pause
classes["pause_short"] = is_short_pause
classes["pause_long"] = is_long_pause
classes["filler"] = is_filler
classes["backchannel"] = is_backchannel
classes["laughter"] = is_laughter
            
def get_prediction_examples(test_data):
    prediction_examples = []
    labels = []
    transcript_marker = "Transcript: "
    for test_example in tqdm(test_data, desc=f"Preparing Examples"):
        prefix, transcript = test_example.split(transcript_marker)
        prefix += transcript_marker
        
        transcript_words = re.split(r"(?<!\[%) ", transcript)
        test_example_labels = []
        for i in range(1, len(transcript_words)):
            ngram = " ".join(transcript_words[i:i+5])
            transcript_history = f"{prefix}{' '.join(transcript_words[:i])}"
            prediction_examples.append(transcript_history)
            pred_example_labels = tuple([criterion(transcript_history, ngram) for criterion in classes.values()])
            test_example_labels.append(pred_example_labels)

        test_example_labels = np.vstack(test_example_labels)

        # classes with no positive instances in this test example should not be evaluated on negative instances from this example
        # because the annotator may not have annotated any instances of this class in the example or even its entire source corpus
        for i in range(test_example_labels.shape[-1]):
            if not np.any(test_example_labels[:,i] == 1):
                test_example_labels[:,i] = -1
        labels.append(test_example_labels)
           
    labels = np.concatenate(labels, axis=0)
    
    # sort examples by length for greater batch efficiency
    prediction_examples = list(zip(prediction_examples, labels))
    prediction_examples.sort(key=lambda x: len(x[0]), reverse=True)
    
    prediction_examples, labels = zip(*prediction_examples)
    labels_df = pd.DataFrame(labels, columns=classes.keys())
    return prediction_examples, labels_df