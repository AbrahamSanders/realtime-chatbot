import torch

def get_device_map():
    num_devices = torch.cuda.device_count()
    # Best case: we have a dedicated device for each model.
    if num_devices > 2:
        tts_device = torch.device("cuda:0")
        asr_device = torch.device("cuda:1")
        agent_device = torch.device("cuda:2")
    # Next best case: we have two devices. Put the agent on its own device 
    # and let ASR & TTS share the other.
    elif num_devices == 2:
        tts_device = asr_device = torch.device("cuda:0")
        agent_device = torch.device("cuda:1")
    # Next best case: we have one device. Put all models on it.
    elif num_devices == 1:
        tts_device = asr_device = agent_device = torch.device("cuda:0")
    # Worst case: Put all models on CPU.
    else:
        tts_device = asr_device = agent_device = torch.device("cpu")
    
    return {
        "tts": tts_device,
        "asr": asr_device,
        "agent": agent_device
    }