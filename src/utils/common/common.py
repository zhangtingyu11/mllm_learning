import random
import numpy as np
import torch

def set_random_seed(seed: int):
    """
    设置全局随机种子(支持PyTorch/Numpy/Python)
    
    参数:
        seed (int): 随机种子
        deterministic (bool): 是否启用PyTorch确定性模式(更严格但可能降低性能)
    """
    # 基础随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch相关设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    # 启用确定性模式（可选）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

