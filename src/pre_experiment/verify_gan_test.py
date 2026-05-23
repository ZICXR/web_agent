"""
使用训练好的 CGAN 判别器在 browsesafe_bench test 集上进行评估。
输出分类报告、混淆矩阵，并保存预测结果。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json

from utils.gan import Discriminator

# ========================= 全局配置 =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "browsesafe_bench")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "gan")
CACHE_PATH = os.path.join(PROJECT_ROOT, "datasets", "embeddings_cache_test.pt")
CKPT_PATH = os.path.join(RESULTS_DIR, "checkpoints", "discriminator.pt")
ENCODER_NAME = "answerdotai/ModernBERT-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 512
BATCH_SIZE = 256

LABEL_MAP = {"no": 0, "yes": 1}
LABEL_NAMES = {0: "Normal", 1: "Injected"}


# ==================== 特征提取与缓存 =======================

def extract_test_embeddings(encoder_name: str = ENCODER_NAME):
    """
    提取 test 集的 768 维嵌入向量，支持本地缓存。
    """
    if os.path.exists(CACHE_PATH):
        print(f"[缓存] 发现 {CACHE_PATH}，直接加载...")
        cache = torch.load(CACHE_PATH, weights_only=True)
        return cache["embeddings"], cache["labels"]

    ds = load_from_disk(DATASET_DIR)["test"]
    print(f"[数据] test 集: {len(ds)} 条")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(DEVICE)
    model.eval()

    embeddings = []
    labels = []

    for i in tqdm(range(len(ds)), desc="Encoding test"):
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

    torch.save({"embeddings": embeddings, "labels": labels}, CACHE_PATH)
    print(f"[缓存] 已保存至 {CACHE_PATH}")

    return embeddings, labels


# ======================== 评估函数 =========================

def evaluate(discriminator, embeddings, labels):
    """
    在测试集上批量预测，返回每条样本的预测结果。
    """
    discriminator.eval()
    dataset = torch.utils.data.TensorDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for emb, lbl in tqdm(dataloader, desc="Predicting"):
            emb = emb.to(DEVICE)
            class_logits, _ = discriminator(emb)       # 只用分类头
            prob = torch.softmax(class_logits, dim=1)   # (batch, 2)

            preds = prob.argmax(dim=1)
            all_probs.append(prob.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(lbl)

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return all_probs, all_preds, all_labels


# ======================== 报告生成 =========================

def print_report(all_labels, all_preds, all_probs):
    """打印分类报告和混淆矩阵。"""
    # 统计
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 50)
    print("CGAN 判别器 — Test 集评估报告")
    print("=" * 50)
    print(f"样本总数:      {len(all_labels)}")
    print(f"  Normal (0):  {(all_labels == 0).sum()}")
    print(f"  Injected(1): {(all_labels == 1).sum()}")
    print()
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print()
    print("混淆矩阵:")
    print(f"              Pred_Normal  Pred_Injected")
    print(f"  Real_Normal     {tn:>5d}        {fp:>5d}")
    print(f"  Real_Injected   {fn:>5d}        {tp:>5d}")

    return {"accuracy": float(acc), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def save_results(all_probs, all_preds, all_labels, metrics):
    """保存详细预测结果和指标。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 保存 JSON 结果
    results = {
        "metrics": metrics,
        "predictions": [],
    }
    for i in range(len(all_preds)):
        results["predictions"].append({
            "index": i,
            "true_label": int(all_labels[i]),
            "true_name": LABEL_NAMES[all_labels[i]],
            "pred_label": int(all_preds[i]),
            "pred_name": LABEL_NAMES[all_preds[i]],
            "prob_normal": round(float(all_probs[i][0]), 6),
            "prob_injected": round(float(all_probs[i][1]), 6),
        })

    json_path = os.path.join(RESULTS_DIR, "test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 预测结果 -> {json_path}")

    # 保存混淆矩阵图
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array([[metrics["confusion_matrix"]["tn"], metrics["confusion_matrix"]["fp"]],
                    [metrics["confusion_matrix"]["fn"], metrics["confusion_matrix"]["tp"]]])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Injected"])
    ax.set_yticklabels(["Real Normal", "Real Injected"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title(f"Confusion Matrix (Acc={metrics['accuracy']:.4f})")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[保存] 混淆矩阵 -> {cm_path}")

    # 保存概率分布直方图
    normal_probs = all_probs[all_labels == 0, 1]
    injected_probs = all_probs[all_labels == 1, 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_probs, bins=50, alpha=0.6, label="Normal (label=0)", color="steelblue")
    ax.hist(injected_probs, bins=50, alpha=0.6, label="Injected (label=1)", color="tomato")
    ax.set_xlabel("P(Injected)")
    ax.set_ylabel("Count")
    ax.set_title("Test Set — P(Injected) Distribution")
    ax.legend()
    plt.tight_layout()
    hist_path = os.path.join(RESULTS_DIR, "test_prob_distribution.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"[保存] 概率分布 -> {hist_path}")


# ======================== 主入口 =========================

if __name__ == "__main__":
    print(f"[设备] {DEVICE}")

    # 1. 提取 test 集嵌入
    embeddings, labels = extract_test_embeddings()
    print(f"[数据] 维度: {embeddings.shape}, Normal: {sum(labels==0).item()}, Injected: {sum(labels==1).item()}")

    # 2. 加载训练好的判别器
    D = Discriminator(embedding_dim=768).to(DEVICE)
    D.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    print(f"[模型] 已加载 {CKPT_PATH}")

    # 3. 评估
    all_probs, all_preds, all_labels = evaluate(D, embeddings, labels)

    # 4. 报告
    metrics = print_report(all_labels, all_preds, all_probs)

    # 5. 保存结果
    save_results(all_probs, all_preds, all_labels, metrics)
