import hydra
import logging
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from src.utils.common.logger import setup_logging

@hydra.main(config_path="../configs", config_name="config", version_base=None)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    logger.info("Load Model on Device: {}".format(device))
    
    pretrain_optimizer = hydra.utils.instantiate(cfg.pretraining.optimizer, params=model.parameters())
    train_dataset = hydra.utils.instantiate(
        cfg.dataset,
        train=True
    )
    val_dataset = hydra.utils.instantiate(
        cfg.dataset,
        train=False
    )
    train_dataloader = hydra.utils.instantiate(
        cfg.pretraining.train_dataloader,
        dataset = train_dataset
    )
    val_dataloader = hydra.utils.instantiate(
        cfg.pretraining.val_dataloader,
        dataset = val_dataset
    )
    epochs = cfg.pretraining.epochs
    
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_progress_bar = tqdm(train_dataloader, desc=f'Train Epoch [{epoch+1}/{epochs}]', leave=True)
        for idx, ((img1, img2), _) in enumerate(train_progress_bar):
            img1, img2 = img1.to(device), img2.to(device)
            loss = model(img1, img2)
            loss.backward()
            pretrain_optimizer.step()
            pretrain_optimizer.zero_grad()
            
            # 更新进度条描述（替代原有print）
            if idx % 10 == 0:
                train_progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos_sim': f'{model.l_pos.mean():.3f}',
                    'neg_sim': f'{model.l_neg.mean():.3f}'
                })
        
        # 验证
        model.eval()
        val_progress_bar = tqdm(val_dataloader, desc=f'Val Epoch [{epoch+1}/{epochs}]', leave=True)
        for idx, ((img1, img2), _) in enumerate(val_progress_bar):
            img1, img2 = img1.to(device), img2.to(device)
            loss = model(img1, img2)
            if idx % 100 == 0:
                val_progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos_sim': f'{model.l_pos.mean():.3f}',
                    'neg_sim': f'{model.l_neg.mean():.3f}'
                })

    logger.info("Training completed.")

if __name__ == "__main__":
    main()
