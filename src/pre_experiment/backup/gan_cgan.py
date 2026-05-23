"""
条件生成对抗网络 (AC-GAN 变体) 模型定义
用于隐空间下的 HTML 内容注入检测

改进点：
- 判别器双头输出：source (真/假) + class (正常/注入)
- 共享特征层使用 SpectralNorm 增强稳定性
- 分类头与真假头解耦，避免对抗训练削弱分类能力
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    """
    生成器 G：噪声 + 条件标签 -> 伪造 768 维嵌入。
    """

    def __init__(self, latent_dim: int = 100, embedding_dim: int = 768, label_embed_dim: int = 50):
        super().__init__()
        self.label_embed = nn.Embedding(2, label_embed_dim)
        input_dim = latent_dim + label_embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, embedding_dim),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_embed(labels)
        x = torch.cat([z, c], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """
    AC-GAN 判别器（双头架构）：
    - 共享特征提取层 -> 分类头 (2类: 正常/注入) + 真假头 (2类: 真/假)
    - 预测时只用分类头，不受对抗训练干扰

    forward(x) -> (class_logits, source_logits)
      class_logits:  (batch, 2) — 0=Normal, 1=Injected
      source_logits: (batch, 2) — 0=Real, 1=Fake
    """

    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        # 共享特征层
        self.shared = nn.Sequential(
            spectral_norm(nn.Linear(embedding_dim, 1024)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        # 分类头：Normal / Injected
        self.class_head = nn.Linear(256, 2)
        # 真假头：Real / Fake
        self.source_head = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor):
        feat = self.shared(x)
        class_logits = self.class_head(feat)
        source_logits = self.source_head(feat)
        return class_logits, source_logits
