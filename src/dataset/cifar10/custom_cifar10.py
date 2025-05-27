# src/dataset/custom_cifar10.py
from torchvision.datasets import CIFAR10
from omegaconf import DictConfig
from hydra.utils import instantiate

class CustomCIFAR10(CIFAR10):
    """支持预训练/微调阶段和训练/验证模式自动切换transform的数据集类"""
    def __init__(
        self,
        root: str,
        train: bool = True,
        stage: str = "pretraining",  # pretraining/finetuning
        transforms_cfg: DictConfig = None,
        download: bool = False,
        **kwargs
    ):
        # 先初始化父类（transform设为None，稍后处理）
        super().__init__(
            root=root,
            train=train,
            transform=None,
            download=download,
            **kwargs
        )
        
        # 验证配置有效性
        if not hasattr(transforms_cfg, stage):
            raise ValueError(f"Missing transform config for stage: {stage}")
        
        # 根据阶段和模式选择transform配置
        mode = "train" if train else "val"
        target_transform = getattr(transforms_cfg, stage).get(mode)
        
        # 实例化transform
        self.transform = target_transform
        self.stage = stage
        self.mode = mode

    def __repr__(self):
        return f"StageAwareCIFAR10(stage={self.stage}, mode={self.mode}, transform={self.transform})"