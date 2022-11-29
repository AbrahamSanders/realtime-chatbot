def join_queue(queue, delim=" "):
    next_value = []
    while not queue.empty():
        next_value.append(queue.get())
    return delim.join(next_value)

def transfer_queue(source, target):
    while not source.empty():
        target.put(source.get())