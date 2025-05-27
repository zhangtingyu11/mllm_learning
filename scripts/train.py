import hydra
import logging
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path
from src.utils.common.logger import setup_logging

def pretrain(cfg: DictConfig, device: str, logger: logging.Logger) -> nn.Module:
    """预训练模型"""
    logger.info("Starting Pre-training Phase")
    ckpt_save_folder = Path(f"checkpoint/pretrain/{cfg.model._target_}_{cfg.dataset._target_}")
    ckpt_save_folder.mkdir(parents=True, exist_ok=True)
    
    # 初始化模型
    model = hydra.utils.instantiate(cfg.model)
    model.to(device).train()
    logger.info(f"Load Model on Device: {device}")
    
    # 初始化优化器和数据加载器
    optimizer = hydra.utils.instantiate(cfg.pretraining.optimizer, params=model.parameters())
    train_dataset = hydra.utils.instantiate(cfg.dataset, transforms=cfg.dataset.transforms.pretraining.train, train=True)
    val_dataset = hydra.utils.instantiate(cfg.dataset, transforms=cfg.dataset.transforms.pretraining.val, train=False)
    train_dataloader = hydra.utils.instantiate(cfg.pretraining.train_dataloader, dataset=train_dataset)
    val_dataloader = hydra.utils.instantiate(cfg.pretraining.val_dataloader, dataset=val_dataset)
    
    # 预训练循环
    for epoch in range(cfg.pretraining.epochs):
        model.train()
        train_progress_bar = tqdm(train_dataloader, desc=f'Train Epoch [{epoch+1}/{cfg.pretraining.epochs}]')
        for idx, ((img1, img2), _) in enumerate(train_progress_bar):
            img1, img2 = img1.to(device), img2.to(device)
            loss = model(img1, img2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if idx % 10 == 0:
                train_progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos_sim': f'{model.l_pos.mean():.3f}',
                    'neg_sim': f'{model.l_neg.mean():.3f}'
                })
        
        logger.info(f"Epoch [{epoch+1}/{cfg.pretraining.epochs}], "
                   f"Train Loss: {loss.item():.4f}, "
                   f"Pos Sim: {model.l_pos.mean():.3f}, "
                   f"Neg Sim: {model.l_neg.mean():.3f}")
        
        if epoch % 10 == 9:
            # 验证
            model.eval()
            val_progress_bar = tqdm(val_dataloader, desc=f'Val Epoch [{epoch+1}/{cfg.pretraining.epochs}]')
            for idx, ((img1, img2), _) in enumerate(val_progress_bar):
                img1, img2 = img1.to(device), img2.to(device)
                loss = model(img1, img2)
                
                if idx % 100 == 0:
                    val_progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'pos_sim': f'{model.l_pos.mean():.3f}',
                        'neg_sim': f'{model.l_neg.mean():.3f}'
                    })
            
            logger.info(f"Epoch [{epoch+1}/{cfg.pretraining.epochs}], "
                       f"Val Loss: {loss.item():.4f}, "
                       f"Pos Sim: {model.l_pos.mean():.3f}, "
                       f"Neg Sim: {model.l_neg.mean():.3f}")
            
            torch.save(model.state_dict(), f"{str(ckpt_save_folder)}/epoch_{epoch+1}.pth")
    
    logger.info("Pre-training completed.")
    return model

def finetune(cfg: DictConfig, device: str, logger: logging.Logger, pretrained_path: str) -> nn.Module:
    """微调预训练模型"""
    logger.info("Starting Fine-tuning Phase")
    
    # 加载预训练模型
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    logger.info(f"Loaded pre-trained model from {pretrained_path}")
    ckpt_save_folder = Path(f"checkpoint/finetune/{cfg.model._target_}_{cfg.dataset._target_}")
    ckpt_save_folder.mkdir(parents=True, exist_ok=True)
    # 修改模型结构用于分类
    model_fc = model.encoder_q.get_fc_layer()
    model_fc = nn.Linear(model_fc.in_features, 10)
    nn.init.kaiming_normal_(model_fc.weight, mode='fan_in')
    nn.init.zeros_(model_fc.bias)
    model = model.to(device)
    
    # 初始化优化器和数据加载器
    optimizer = hydra.utils.instantiate(cfg.finetuning.optimizer, params=model.parameters())
    train_dataset = hydra.utils.instantiate(cfg.dataset, stage="finetuning", train=True)
    val_dataset = hydra.utils.instantiate(cfg.dataset, stage="finetuning", train=False)
    train_dataloader = hydra.utils.instantiate(cfg.finetuning.train_dataloader, dataset=train_dataset)
    val_dataloader = hydra.utils.instantiate(cfg.finetuning.val_dataloader, dataset=val_dataset)
    
    # 微调循环
    for epoch in range(cfg.finetuning.epochs):
        model.train()
        train_progress_bar = tqdm(train_dataloader, desc=f'Fine-tune Epoch [{epoch+1}/{cfg.finetuning.epochs}]')
        for idx, (images, labels) in enumerate(train_progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / labels.size(0)
            
            if idx % 10 == 0:
                train_progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.2f}%'
                })
        
        logger.info(f"Epoch [{epoch+1}/{cfg.finetuning.epochs}], "
                   f"Train Loss: {loss.item():.4f}, "
                   f"Train Acc: {accuracy:.2f}%")
        

        
        if epoch % 10 == 9:
            # 验证
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_progress_bar = tqdm(val_dataloader, desc=f'Val Epoch [{epoch+1}/{cfg.finetuning.epochs}]')
                for idx, (images, labels) in enumerate(val_progress_bar):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    if idx % 100 == 0:
                        val_progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{100 * val_correct / val_total:.2f}%'
                        })
            
            val_accuracy = 100 * val_correct / val_total
            logger.info(f"Epoch [{epoch+1}/{cfg.finetuning.epochs}], "
                       f"Val Loss: {val_loss/len(val_dataloader):.4f}, "
                       f"Val Acc: {val_accuracy:.2f}%")
            
            torch.save(model.state_dict(), f"{str(ckpt_save_folder)}/epoch_{epoch+1}.pth")
    
    logger.info("Fine-tuning completed.")

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    setup_logging(
        log_dir="logs",
        console_level="INFO",
        file_level="DEBUG"
    )
    logger = logging.getLogger(__name__)
    logger.info("Loading configuration...")
    
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 执行预训练
    # pretrain(cfg, device, logger)
    
    # 执行微调
    pretrained_path = "/home/zty/Project/mllm_learning/checkpoint/pretrain/src.model.MocoV1_torchvision.datasets.CIFAR10/epoch_200.pth"
    finetune(cfg, device, logger, pretrained_path)

if __name__ == "__main__":
    main()