import torch 
import logging

logger = logging.getLogger(__name__)

def get_num_gpus():
    num_gpus = torch.cuda.device_count()
    return num_gpus