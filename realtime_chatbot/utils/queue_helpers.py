import numpy as np
from time import sleep

def join_queue(queue, delim=" "):
    buffer = []
    transfer_queue_to_buffer(queue, buffer)
    return delim.join(buffer)

def join_queue_audio(queue):
    buffer = []
    transfer_queue_to_buffer(queue, buffer)
    if len(buffer) > 0:
        return (buffer[0][0], np.concatenate([item[1] for item in buffer]))
    return None

def skip_queue(queue):
    buffer = []
    transfer_queue_to_buffer(queue, buffer)
    return buffer[-1] if len(buffer) > 0 else None

def transfer_queue(source, target):
    buffer = []
    transfer_queue_to_buffer(source, buffer)
    transfer_buffer_to_queue(buffer, target)

def transfer_queue_to_buffer(source, buffer, retries=3, max_read_count=100):
    count = 0
    retry = -1
    while True:
        while not source.empty():
            buffer.append(source.get())
            count += 1
            # we received something so reset the retry count
            retry = -1
            if count == max_read_count:
                break
        # increment the retry count and check for stopping condition
        retry += 1
        if count == 0 or count == max_read_count or retry == retries:
            break
        sleep(0.001 if retry==0 else 0.01)
    return count

def transfer_buffer_to_queue(buffer, target):
    for item in buffer:
        target.put(item)
    buffer.clear()