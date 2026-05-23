import torch
print(torch.__version__)           # 输出: 2.12.0 (无 +cu 后缀)
print(torch.cuda.is_available())   # 输出: False → 确认是 CPU 版