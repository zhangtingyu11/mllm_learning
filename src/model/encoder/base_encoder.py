from abc import ABC, abstractmethod
import torch.nn as nn

class BaseEncoder(ABC, nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
    
    def get_fc_layer(self):
        pass
    
    @abstractmethod
    def forward(self, x):
        pass
