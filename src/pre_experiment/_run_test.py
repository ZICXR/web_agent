"""测试分类器：加载已训练模型，在测试集上评估并输出报告和ROC曲线。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from src.pre_experiment.verify_classifier import (
    MLPClassifier,
    extract_embeddings,
    RESULTS_DIR,
    DEVICE,
    CACHE_TEST,
)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"[设备] {DEVICE}")

# 加载模型
model = MLPClassifier().to(DEVICE)
model.load_state_dict(
    torch.load(
        os.path.join(RESULTS_DIR, "checkpoints", "classifier.pt"),
        weights_only=True,
        map_location=DEVICE,
    )
)
model.eval()
print("[模型] 已加载 classifier.pt")

# 提取测试集嵌入
test_emb, test_labels = extract_embeddings("test", CACHE_TEST)
print(f"[测试集] Normal: {sum(test_labels == 0).item()}, Injected: {sum(test_labels == 1).item()}")

# 推理
with torch.no_grad():
    logits = model(test_emb.to(DEVICE))
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    y_true = test_labels.numpy()

print("\n[分类报告]")
print(classification_report(y_true, preds, target_names=["Normal", "Injected"], digits=4))

cm = confusion_matrix(y_true, preds)
print("[混淆矩阵]")
print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}")
print(f"  FN={cm[1, 0]}  TP={cm[1, 1]}")

auc = roc_auc_score(y_true, probs)
print(f"\nAUC-ROC: {auc:.4f}")

# 绘制 ROC 曲线
fpr, tpr, _ = roc_curve(y_true, probs)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="royalblue")
ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - MLP Classifier")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
plt.close()
print(f"[保存] ROC曲线 -> {RESULTS_DIR}/roc_curve.png")
