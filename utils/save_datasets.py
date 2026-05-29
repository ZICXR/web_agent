# import os
# # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from datasets import load_dataset
# ds = load_dataset("perplexity-ai/browsesafe-bench")

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_info

# 开启下载日志
set_verbosity_info()

SAVE_DIR = os.path.join(
    "..",
    "datasets",
    "browsesafe_bench"
)

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Saving dataset to: {SAVE_DIR}")

# 下载数据集
ds = load_dataset(
    "perplexity-ai/browsesafe-bench"
)

print("Download finished.")

# 保存到本地
ds.save_to_disk(SAVE_DIR)

print(f"Dataset saved to: {SAVE_DIR}")