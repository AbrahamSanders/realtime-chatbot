def join_queue(queue, delim=" "):
    next_value = []
    while not queue.empty():
        next_value.append(queue.get())
    return delim.join(next_value)