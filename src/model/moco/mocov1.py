from .moco import Moco
import torch.nn as nn

class MocoV1(Moco):
    def __init__(self,
                 base_encoder, 
                 queue_size=65536, 
                 momentum=0.999, 
                 temperature=0.07):
        super().__init__(base_encoder, queue_size, momentum, temperature)
