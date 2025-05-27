import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Moco(nn.Module):
    def __init__(self, 
                 base_encoder, 
                 queue_size=65536, 
                 momentum=0.999, 
                 temperature=0.07):
        super(Moco, self).__init__()
        
        self.queue_size = queue_size # 队列大小
        self.momentum = momentum    # 动量更新的动量
        self.temperature = temperature  # infoNCE的温度系数

        # 传进来的是已经参数初始化后的encoder
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)

        # 复制encoder_q的参数到encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # 输出所有模型的名字
        self.register_buffer("queue", torch.randn(base_encoder.get_fc_layer().out_features, self.queue_size))
        # 因为后面要计算余弦相似度，需要先进行归一化
        self.queue = F.normalize(self.queue, dim=0)
        # 队列的起始指针, 初始指针为0
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.iter = 0

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + (param_q.data) * (1 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """出队入队

        Args:
            keys (_type_): 尺寸大小是(batch_size, dim) 
        """
        ptr = int(self.queue_ptr)
        batch_size = keys.shape[0]
        # 入队
        self.queue[:, ptr:ptr+batch_size] = keys.T
        # 移动指针，当相于出队
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, img_q, img_k=None, mode="loss"):
        """前向过程

        Args:
            img_q (_type_): query图片
            img_k (_type_): key图片
        """
        q = self.encoder_q(img_q)
        q = F.normalize(q, dim=1)
        
        if img_k is None:
            return q
        
        with torch.no_grad():
            # 先更新编码器，再求key的特征
            if self.training:
                self._momentum_update_key_encoder()
            k = self.encoder_k(img_k)
            k = F.normalize(k, dim=1)
        if mode == "loss":
            # 相当于先做q * k, 对特征维度求和, 也就是batch_size个样本对应做内积(注意不是两两内积)
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(1)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            self.l_pos = l_pos
            self.l_neg = l_neg
            # 将正负样本的相似度拼接起来
            logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(img_q.device)
            # infoNCE其实就是cross entropy loss
            loss = F.cross_entropy(logits, labels)
            
            # 更新队列
            if self.training:
                self._dequeue_and_enqueue(k)
            return loss
        else:
            raise NotImplementedError
