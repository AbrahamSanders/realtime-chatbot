import re

from realtime_chatbot.realtime_agent import RealtimeAgentConfig

class Test:
    def __init__(self, current_speaker, sequence, partial_pos):
        self.current_speaker = current_speaker
        self.sequence = sequence
        self.partial_pos = partial_pos
        self.any_identity_regex = re.compile(r"S\d+?")
        self.input_segments_regex = re.compile(" (?=[~*])")
        self.sequence_split_regex = re.compile(rf"\s(?={self.any_identity_regex.pattern})")
        self.config = RealtimeAgentConfig()

    def _set_current_speaker(self, identity):
        self.sequence += f" {identity}:"
        self.current_speaker = identity

    def _update_sequence_from_input(self, next_input):
        sequence_changed = False
        if next_input:
            # First, clear out the previous partial utterance segment (if exists)
            next_turn = None
            if self.partial_pos > -1:
                # Locate any turn taken by another speaker after the partial utterance has started
                partial_utterance = self.sequence[self.partial_pos:]
                turn_switch = re.search(self.any_identity_regex, partial_utterance)
                if turn_switch:
                    next_turn = partial_utterance[turn_switch.start():]
                # Reset the sequence to the position where the partial utterance begins
                self.sequence = self.sequence[:self.partial_pos]
                self.partial_pos = -1
            # If no partial utterance exists and the user is not speaking, set the user
            # as the active speaker
            elif self.current_speaker != self.config.user_identity:
                self._set_current_speaker(self.config.user_identity)
            # Next, add the new segments to the transcription, 
            # discarding intermediate partial segments.
            new_segments = re.split(self.input_segments_regex, next_input)
            for i, seg in enumerate(new_segments):
                if len(seg) > 1 and (seg.startswith("*") or i == len(new_segments)-1):
                    if seg.startswith("~"):
                        self.partial_pos = len(self.sequence)
                    self.sequence += f" {seg[1:]}"
            # Finally, in case a partial utterance was replaced that was followed by a next turn,
            # put that next turn back.
            if next_turn:
                self.sequence += f" {next_turn}"
            sequence_changed = True

        return sequence_changed

    def _get_next_slice_index(self, str, i):
        if i < len(str):
            for pos in range(i, len(str)):
                if str[pos] == " ":
                    return pos
        return len(str)

    def _update_sequence_from_input2(self, next_input):
        if not next_input:
            return False

        # First, clear out the previous partial utterance segment (if exists)
        utterances_after_partial_pos = []
        if self.partial_pos > -1:
            # Locate all turns taken after the partial utterance has started
            utterances_after_partial_pos = re.split(self.sequence_split_regex, self.sequence[self.partial_pos:])
            # Reset the sequence to the position where the partial utterance begins
            self.sequence = self.sequence[:self.partial_pos]
            self.partial_pos = -1
        # Next, add the new segments to the sequence, discarding intermediate partial segments.
        new_segments = re.split(self.input_segments_regex, next_input)
        for i, seg in enumerate(new_segments):
            if len(seg) > 1 and (seg.startswith("*") or i == len(new_segments)-1):
                if seg.startswith("~"):
                    self.partial_pos = len(self.sequence)
                seg_text = seg[1:]
                # Iterate through all turns taken after the previous partial utterance started.
                # Replace user utterances with words from the new segment while carrying non-user 
                # utterances over intact.
                for utt in utterances_after_partial_pos:
                    if utt:
                        identity_match = re.match(self.any_identity_regex, utt)
                        if identity_match and not utt.startswith(self.config.user_identity):
                            # carry non-user utterance over intact
                            self._set_current_speaker(identity_match[0])
                            utt = utt[identity_match.end()+1:].lstrip()
                            self.sequence += f" {utt}"
                        elif seg_text:
                            # replace user utterance with words (of same approximate length)
                            # from the new segment
                            has_user_identity = utt.startswith(self.config.user_identity)
                            if has_user_identity:
                                utt = utt[len(self.config.user_identity)+1:]
                            utt = utt.lstrip()
                            next_slice_idx = self._get_next_slice_index(seg_text, len(utt))
                            if has_user_identity:
                                self._set_current_speaker(self.config.user_identity)
                            self.sequence += f" {seg_text[:next_slice_idx]}"
                            seg_text = seg_text[next_slice_idx:].lstrip()
                utterances_after_partial_pos.clear()
                # any remaining text in the new segment is appended to the end of the sequence
                if seg_text:
                    if self.current_speaker != self.config.user_identity:
                        self._set_current_speaker(self.config.user_identity)
                    self.sequence += f" {seg_text}"
        return True

if __name__ == "__main__":
    current_speaker = "S2"
    sequence = "because I've been getting a lot of pains (4.4) S2: (1.7)"
    partial_pos = -1
    input = "*alright yeah I said"

    test = Test(current_speaker, sequence, partial_pos)
    test._update_sequence_from_input(input)
    print("Original:")
    print(test.sequence)
    print(test.partial_pos)
    print(test.current_speaker)

    test2 = Test(current_speaker, sequence, partial_pos)
    test2._update_sequence_from_input2(input)
    print("\nNew:")
    print(test2.sequence)
    print(test2.partial_pos)
    print(test2.current_speaker)