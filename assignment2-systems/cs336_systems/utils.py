import torch
import torch.nn as nn

def get_device(device:str|None)->str:
    if device is not None:
        return device
    elif torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'