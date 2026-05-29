"""
分块编码 (Chunked Encoding) + MLP 分类器
对超长 HTML 文本按 8192 tokens 分块，每块独立提取 CLS 向量，
再通过 Max Pooling 聚合为单个 768 维向量，训练分类器。
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

# ========================= 全局配置 =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "classifier")
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "browsesafe_bench")
ENCODER_NAME = "answerdotai/ModernBERT-base"

# 分块编码使用独立缓存文件，不影响旧版 512 缓存
CACHE_TRAIN = os.path.join(PROJECT_ROOT, "datasets", "embeddings_cache_chunked.pt")
CACHE_TEST = os.path.join(PROJECT_ROOT, "datasets", "embeddings_cache_test_chunked.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 8192       # ModernBERT 最大长度
STRIDE = 6144           # 重叠步长 (8192 * 0.75)，避免恶意内容被截断在边界
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3

LABEL_MAP = {"no": 0, "yes": 1}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==================== 分块编码与缓存 =======================

def encode_single(text, tokenizer, encoder):
    """编码单条文本（短文本直接编码，长文本分块 Max Pooling）。"""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    total_len = len(tokens)

    if total_len <= CHUNK_SIZE:
        inputs = tokenizer(
            text, max_length=CHUNK_SIZE, truncation=True,
            padding="max_length", return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            vec = encoder(**inputs).last_hidden_state[:, 0, :]
        return vec.cpu().squeeze(0)
    else:
        chunk_vecs = []
        for start in range(0, total_len, STRIDE):
            chunk_tokens = tokens[start : start + CHUNK_SIZE]
            if len(chunk_tokens) < 64:
                break
            input_ids = torch.tensor([chunk_tokens], dtype=torch.long).to(DEVICE)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                vec = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            chunk_vecs.append(vec.cpu())

        if chunk_vecs:
            return torch.cat(chunk_vecs, dim=0).max(dim=0).values
        else:
            return torch.zeros(768)


def extract_embeddings(split, cache_path):
    """提取嵌入，支持断点续跑：每 1000 条自动保存，崩溃后从上次位置继续。"""
    labels_t = None

    # 检查是否有未完成的断点缓存
    checkpoint_path = cache_path + ".partial"
    start_idx = 0

    if os.path.exists(checkpoint_path):
        print(f"[断点续跑] 发现 {checkpoint_path}，继续...")
        ckpt = torch.load(checkpoint_path, weights_only=True)
        all_embeddings = [ckpt["embeddings"]]
        start_idx = ckpt["next_idx"]
        labels_t = ckpt["labels"]
        print(f"  已有 {start_idx} 条，从第 {start_idx + 1} 条继续")
    elif os.path.exists(cache_path):
        print(f"[缓存] 发现 {cache_path}，直接加载...")
        cache = torch.load(cache_path, weights_only=True)
        return cache["embeddings"], cache["labels"]
    else:
        all_embeddings = []
        start_idx = 0

    print(f"[编码] 加载 {split} 集...")
    ds = load_from_disk(DATASET_DIR)[split]
    texts = list(ds["content"])

    if labels_t is None:
        labels = [LABEL_MAP[l] for l in ds["label"]]
        labels_t = torch.tensor(labels)

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)
    encoder.eval()

    for i in tqdm(range(start_idx, len(texts)), desc="Chunked encoding", initial=start_idx, total=len(texts)):
        vec = encode_single(texts[i], tokenizer, encoder)
        all_embeddings.append(vec.unsqueeze(0))

        # 每 1000 条保存断点
        if (i + 1) % 1000 == 0:
            emb_so_far = torch.cat(all_embeddings, dim=0)
            torch.save({"embeddings": emb_so_far, "labels": labels_t, "next_idx": i + 1}, checkpoint_path)
            print(f"  [断点] 已保存 {i + 1}/{len(texts)}")

    # 编码完成，合并并保存最终缓存
    embeddings = torch.cat(all_embeddings, dim=0)
    torch.save({"embeddings": embeddings, "labels": labels_t}, cache_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"[缓存] 已保存至 {cache_path}")
    print(f"  维度: {embeddings.shape}")

    return embeddings, labels_t


# ======================== 数据集 =========================

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ======================== 分类器 =========================

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ======================== 训练 =========================

def train():
    print(f"[设备] {DEVICE}")

    embeddings, labels = extract_embeddings("train", CACHE_TRAIN)
    print(f"[数据] Normal: {sum(labels==0).item()}, Injected: {sum(labels==1).item()}")

    dataset = EmbeddingDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = MLPClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for emb, lbl in pbar:
            emb, lbl = emb.to(DEVICE), lbl.to(DEVICE)
            logits = model(emb)
            loss = criterion(logits, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (logits.argmax(1) == lbl).sum().item()
            total += lbl.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        avg_loss = running_loss / len(dataloader)
        acc = correct / total
        history["loss"].append(avg_loss)
        history["acc"].append(acc)
        print(f"Epoch {epoch:>2d}/{EPOCHS}  loss={avg_loss:.4f}  acc={acc:.4f}")

    return model, history


# ======================== 主入口 =========================

if __name__ == "__main__":
    model, history = train()

    # 保存模型
    model_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "classifier.pt"))
    print(f"\n[保存] 模型 -> {model_dir}/classifier.pt")

    # 保存训练曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(history["loss"], label="Loss", color="tomato")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(history["acc"], label="Accuracy", color="green", linestyle="--")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="upper right")

    plt.title("MLP Classifier on Chunked ModernBERT Embeddings (8192)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "train_curve.png"), dpi=150)
    plt.close()
    print(f"[保存] 训练曲线 -> {RESULTS_DIR}/train_curve.png")
