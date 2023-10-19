def decode_with_prefix_space(tokenizer, token_ids, *args, **kwargs):
    text = tokenizer.decode(token_ids, *args, **kwargs)
    # Make sure a space is prefixed to the generated text if the first generated token starts with '▁'
    # (necessary because different tokenizers handle lead spacing differently)
    if len(token_ids) > 0:
        first_generated_token = tokenizer.convert_ids_to_tokens(token_ids[0:1])[0]
        if first_generated_token.startswith("▁") and not text.startswith(" "):
            text = f" {text}"
    return text

def get_token_id(tokenizer, token):
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if token.startswith(" ") and tokenizer.convert_ids_to_tokens(token_ids[0]) == "▁" and len(token_ids) > 1:
        return token_ids[1]
    return token_ids[0]