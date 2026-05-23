"""
Load perplexity-ai/browsesafe-bench dataset from local disk.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets import load_from_disk

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAVE_DIR = os.path.join(PROJECT_ROOT, "datasets", "browsesafe_bench")


def load_browsesafe():
    print(f"Loading from: {SAVE_DIR}")
    ds = load_from_disk(SAVE_DIR)
    print(f"Splits: {list(ds.keys())}")
    for split, data in ds.items():
        print(f"  {split}: {len(data)} samples, columns: {data.column_names}")
    return ds


if __name__ == "__main__":
    ds = load_browsesafe()

    sample = ds["train"][0]
    print("\n--- Sample 0 (train) ---")
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"  {key}: {value[:200]}... (len={len(value)})")
        else:
            print(f"  {key}: {value}")
