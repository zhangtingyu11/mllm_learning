import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from src.utils.common.logger import setup_logging

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_logging(
        log_dir="logs",
        console_level="INFO",
        file_level="DEBUG"
    )
    logger = logging.getLogger(__name__)
    logger.info("Loading configuration...")

    # 打印完整配置
    model = hydra.utils.instantiate(cfg.model)
    train_dataset = hydra.utils.instantiate(
        cfg.dataset,
        train=True
    )
    val_dataset = hydra.utils.instantiate(
        cfg.dataset,
        train=False
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.pretraining.batch_size,
        shuffle=True,
        num_workers=cfg.pretraining.num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.pretraining.batch_size,
        shuffle=False,
        num_workers=cfg.pretraining.num_workers
    )

    epochs = cfg.pretraining.epochs
    
    for (img1, img2), _ in train_dataloader:
        print(img1.shape)
        print(img2.shape)
        exit(0)

if __name__ == "__main__":
    main()
