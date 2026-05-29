"""
分块编码 MLP 分类器 — Test 集评估
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np

from src.pre_experiment.verify_classifier import extract_embeddings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "classifier")
CKPT_PATH = os.path.join(RESULTS_DIR, "checkpoints", "classifier.pt")
CACHE_TEST = os.path.join(PROJECT_ROOT, "datasets", "embeddings_cache_test_chunked.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
LABEL_NAMES = {0: "Normal", 1: "Injected"}


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


def evaluate():
    print(f"[设备] {DEVICE}")

    print(f"[缓存] 加载测试集嵌入...")
    embeddings, labels = extract_embeddings("test", CACHE_TEST)
    print(f"  维度: {embeddings.shape}, Normal: {sum(labels==0).item()}, Injected: {sum(labels==1).item()}")

    model = MLPClassifier().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"[模型] 已加载 {CKPT_PATH}")

    dataset = torch.utils.data.TensorDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for emb, lbl in tqdm(dataloader, desc="Predicting"):
            logits = model(emb.to(DEVICE))
            prob = torch.softmax(logits, dim=1).cpu()
            all_probs.append(prob.numpy())
            all_preds.append(prob.argmax(1).numpy())
            all_labels.append(lbl.numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    acc = (tp + tn) / len(all_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 50)
    print("Chunked MLP Classifier — Test 集评估报告")
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

    metrics = {"accuracy": float(acc), "precision": float(precision),
               "recall": float(recall), "f1": float(f1),
               "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}

    results = {"metrics": metrics, "predictions": []}
    for i in range(len(all_preds)):
        results["predictions"].append({
            "index": i,
            "true_label": int(all_labels[i]),
            "pred_label": int(all_preds[i]),
            "prob_normal": round(float(all_probs[i][0]), 6),
            "prob_injected": round(float(all_probs[i][1]), 6),
        })
    json_path = os.path.join(RESULTS_DIR, "test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 预测结果 -> {json_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Injected"])
    ax.set_yticklabels(["Real Normal", "Real Injected"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=14)
    ax.set_title(f"Confusion Matrix (Acc={acc:.4f}, F1={f1:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "test_confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[保存] 混淆矩阵 -> {RESULTS_DIR}/test_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_probs[all_labels==0, 1], bins=50, alpha=0.6, label="Normal", color="steelblue")
    ax.hist(all_probs[all_labels==1, 1], bins=50, alpha=0.6, label="Injected", color="tomato")
    ax.set_xlabel("P(Injected)")
    ax.set_ylabel("Count")
    ax.set_title("Test Set — P(Injected) Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "test_prob_distribution.png"), dpi=150)
    plt.close()
    print(f"[保存] 概率分布 -> {RESULTS_DIR}/test_prob_distribution.png")


if __name__ == "__main__":
    evaluate()
