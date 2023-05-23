import re

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

    num_responses_with_common_responses = 0
    for response in response_list:
        # normalize response
        response = response.lower()
        response = response.replace("really", "")
        response = response.replace("very", "")
        response = response.replace("so", "")
        response = response.replace("do not", "don't")
        response = response.replace("cannot", "can't")
        response = response.replace("can not", "can't")
        response = response.replace("i am", "i'm")
        response = re.sub(" {2,}", " ", response)
        response = response.strip()

        # check if response contains common response
        for common_response in common_responses:
            if common_response in response:
                num_responses_with_common_responses += 1
                break
    proportion_with_common = num_responses_with_common_responses / len(response_list)
    return proportion_with_common