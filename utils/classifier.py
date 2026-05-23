"""
基于 ModernBERT 的网页注入检测分类器。
在预训练编码器顶部添加分类头，端到端微调。
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class WebInjectionClassifier(nn.Module):
    """
    ModernBERT + 分类头，用于判断网页是否被注入。
    - 输入：tokenized HTML 文本
    - 输出：2 类 logits (0=Normal, 1=Injected)
    """

    def __init__(self, encoder_name: str = "answerdotai/ModernBERT-base", freeze_layers: int = 10):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)

        # 冻结底层 transformer 层，只微调顶层 + 分类头
        if freeze_layers > 0:
            # embeddings 层冻结
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            # 冻结前 N 层
            for i, layer in enumerate(self.encoder.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        hidden_dim = self.encoder.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return self.classifier(cls_vec)
