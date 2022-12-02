import numpy as np

def join_queue(queue, delim=" "):
    next_value = []
    while not queue.empty():
        next_value.append(queue.get())
    return delim.join(next_value)

def join_queue_audio(queue):
    next_value = []
    while not queue.empty():
        next_value.append(queue.get())
    if len(next_value) > 0:
        return (next_value[0][0], np.concatenate([val[1] for val in next_value]))
    return None

def skip_queue(queue):
    last_value = None
    while not queue.empty():
        last_value = queue.get()
    return last_value

def transfer_queue(source, target):
    while not source.empty():
        target.put(source.get())