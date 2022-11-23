import pylangacq
import re
from tqdm import tqdm

class TalkbankDataLoader:
    def __init__(self, max_utterance_words=150, max_example_words=500, min_overlap_words=100):
        self.max_utterance_words = max_utterance_words
        self.max_example_words = max_example_words
        self.min_overlap_words = min_overlap_words
        self.corpora_urls = {
            "CABNC": "https://ca.talkbank.org/data/CABNC.zip",
            "CallFriend_eng_n": "https://ca.talkbank.org/data/CallFriend/eng-n.zip",
            "CallFriend_eng_s": "https://ca.talkbank.org/data/CallFriend/eng-s.zip",
            "CallHome_eng": "https://ca.talkbank.org/data/CallHome/eng.zip",
            "GCSAusE": "https://ca.talkbank.org/data/GCSAusE.zip",
            "ISL": "https://ca.talkbank.org/data/ISL.zip",
            "MICASE": "https://ca.talkbank.org/data/MICASE.zip",
            "SCoSE": "https://ca.talkbank.org/data/SCoSE.zip"
        }
    
    def clean_line(self, line):
        # convert 'hello [!]' to 'hello!'
        line = re.sub(r" \[!\]", "!", line)
        # get rid of bracketed sequences that don't contain a comment or sound
        line = re.sub(r"\[[^%\]].*?\]", "", line)
        # get rid of timestamp
        line = re.sub(r"\d+?_\d+?", "", line)
        # get rid of +" and +,
        line = re.sub(r'\+[",]', "", line)
        # get rid of +/.
        line = re.sub(r'\+/\.', "", line)
        # get rid of &- (comes before fillers, e.g., 'um')
        line = re.sub("&-", "", line)
        # replace ° or ☺ with a single space
        line = re.sub("[°☺]", " ", line)
        # get rid of any non word or non punctuation characters
        line = re.sub(r"[^\w !?.,;\"'`()&=%\[\]]", "", line)
        # get rid of ʔ which is somehow a word character 
        line = re.sub("ʔ", "", line)
        # normalize sequences of spaces to a single space
        line = re.sub(" {2,}", " ", line)
        # close punctuation and contractions that have an extra space
        # between the word and the punctuation or contraction
        line = re.sub(" (?=[!?.,;'])", "", line)
        line = re.sub(" (?=n')", "", line)
        # finally, strip the line
        line = line.strip()
        return line
    
    def get_participants_str(self, header):
        participants = header["Participants"]
        part_str = ""
        part_map = {}
        for i, item in enumerate(participants.items()):
            part, info = item
            part_map[part] = f"S{i+1}"
            part_str += f"<participant> {part_map[part]} ("
            for info_key in ("name", "age", "sex"):
                default_value = "unknown"
                value = info[info_key]
                if info_key == "name" and value.lower() in ["male", "female"]:
                    info["sex"] = value.lower()
                    value = default_value
                if info_key == "age" and ";" in value:
                    value = value[:value.index(";")]
                part_str += f"{info_key}: {value if value else default_value}, "
            part_str = part_str.rstrip(", ")
            part_str += ") "
        return part_str.rstrip(), part_map
    
    def get_utterances_str(self, utterances, part_map, start):
        cleaned_utts = []
        cleaned_utts_word_counts = []
        total_word_count = 0
        next_start = None
        for i in range(start, len(utterances)):
            utt = utterances[i]
            
            # clean the utterance and prepend the speaker
            clean_utt = self.clean_line(utt.tiers[utt.participant])
            clean_utt = f"{part_map[utt.participant]}: {clean_utt}"
            
            # count the words and truncate if too long
            clean_utt_split = clean_utt.split()
            clean_utt_word_count = len(clean_utt_split)
            if clean_utt_word_count > self.max_utterance_words:
                clean_utt = " ".join(clean_utt_split[:self.max_utterance_words])
                clean_utt = f"{clean_utt}..."
                clean_utt_word_count = self.max_utterance_words
            
            # if there is room left in the example append the cleaned utterance to the list.
            # otherwise skip it and end the example here.
            if total_word_count + clean_utt_word_count <= self.max_example_words:
                cleaned_utts.append(clean_utt)
                cleaned_utts_word_counts.append(clean_utt_word_count)
                total_word_count += clean_utt_word_count
            else:
                next_start = i
                # compute how many utterances to backtrack for next_start to achieve the minimum overlap size
                if self.min_overlap_words > 0:
                    overlap_word_count = 0
                    for j in range(len(cleaned_utts_word_counts)-1, -1, -1):
                        overlap_word_count += cleaned_utts_word_counts[j]
                        next_start -= 1
                        if overlap_word_count >= self.min_overlap_words:
                            break
                break
            
        # compile all utterances in the example into a sequence
        utts_str = " ".join(cleaned_utts)
        return utts_str, next_start
    
    def load_data(self, corpora="All", exclude=None):
        if isinstance(corpora, str):
            if corpora == "All":
                corpora = list(self.corpora_urls)
            else:
                corpora = corpora.split(",")
            
        for corpus in corpora:
            if corpus not in self.corpora_urls:
                raise ValueError(f"Corpus '{corpus}' is not currently supported. "
                                 f"Choose from {list(self.corpora_urls)}, passed as a list "
                                 "or a comma delimited string, or pass 'All'.")
        
        for corpus in tqdm(corpora, desc="Corpora"):
            corpus_url = self.corpora_urls[corpus]
            reader = pylangacq.read_chat(corpus_url, exclude=exclude)
            
            all_headers = reader.headers()
            all_utterances = reader.utterances(by_files=True)
            
            for header, utterances in tqdm(zip(all_headers, all_utterances), desc="Files"):
                part_str, part_map = self.get_participants_str(header)
                prefix = f"{part_str} <dialog> "
                
                start = 0
                while start is not None:
                    utts_str, start = self.get_utterances_str(utterances, part_map, start)
                    if len(utts_str) > 0:
                        yield f"{prefix}{utts_str}"