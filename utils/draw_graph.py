import matplotlib.pyplot as plt
import numpy as np

def plot_single_array(arr, title="Single Array Line Plot"):
    plt.figure()
    plt.plot(arr, linestyle='-')  # 折线 + 点
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def plot_two_arrays_same_plot(arr1, arr2, title="Two Arrays Line Plot"):
    plt.figure()
    plt.plot(arr1, linestyle='-', label="Array 1")
    plt.plot(arr2, linestyle='-', label="Array 2")
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_two_arrays_subplots(arr1, arr2, title="Two Arrays Subplots"):
    # 将输入转换为 numpy 数组（便于处理）
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    # 确定最大长度，并对短数组补零
    max_len = max(len(arr1), len(arr2))
    if len(arr1) < max_len:
        arr1 = np.pad(arr1, (0, max_len - len(arr1)), constant_values=0)
    if len(arr2) < max_len:
        arr2 = np.pad(arr2, (0, max_len - len(arr2)), constant_values=0)

    # 创建子图并绘图
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(arr1, linestyle='-')
    axs[0].set_title("Array 1")
    axs[0].grid(True)

    axs[1].plot(arr2, linestyle='-')
    axs[1].set_title("Array 2")
    axs[1].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
