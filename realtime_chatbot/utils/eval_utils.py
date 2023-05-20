def measure_common_responses(response_list):
    # measure the frequency of responses that contain common response phrases
    # e.g., "I don't know", "I'm not sure", etc.
    common_responses = [
        "i don't know",
        "i don't think so",
        "i don't remember",
        "i can't remember",
        "i'm not sure",
        "i'm sorry"
    ]
    common_responses_expanded = []
    for response in common_responses:
        if "don't" in response:
            common_responses_expanded.append(response.replace("don't", "do not"))
        if "can't" in response:
            common_responses_expanded.append(response.replace("can't", "cannot"))
            common_responses_expanded.append(response.replace("can't", "can not"))
        if "i'm" in response:
            common_responses_expanded.append(response.replace("i'm", "i am"))
    common_responses = common_responses + common_responses_expanded

    num_responses_with_common_responses = 0
    for response in response_list:
        response = response.lower()
        for common_response in common_responses:
            if common_response in response:
                num_responses_with_common_responses += 1
                break
    proportion_with_common = num_responses_with_common_responses / len(response_list)
    return proportion_with_common