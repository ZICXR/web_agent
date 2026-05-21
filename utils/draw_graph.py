import matplotlib.pyplot as plt

def plot_single_array(arr, title="Single Array Line Plot"):
    plt.figure()
    plt.plot(arr, linestyle='-', marker='o')  # 折线 + 点
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def plot_two_arrays_same_plot(arr1, arr2, title="Two Arrays Line Plot"):
    plt.figure()
    plt.plot(arr1, linestyle='-', marker='o', label="Array 1")
    plt.plot(arr2, linestyle='-', marker='s', label="Array 2")
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_two_arrays_subplots(arr1, arr2, title="Two Arrays Subplots"):
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(arr1, linestyle='-', marker='o')
    axs[0].set_title("Array 1")
    axs[0].grid(True)

    axs[1].plot(arr2, linestyle='-', marker='s')
    axs[1].set_title("Array 2")
    axs[1].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
