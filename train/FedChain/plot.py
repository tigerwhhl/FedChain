import matplotlib.pyplot as plt

def plot(epochs,loss,acc, save_path="./results/result.png"):
    ax1 = plt.subplot(1, 2, 1)
    plt.sca(ax1)
    plt.plot(epochs,loss, "r", label="train_loss")
    plt.legend()

    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax2)
    plt.plot(epochs, acc, "b", label="train_acc")
    plt.legend()
    plt.savefig(save_path)

if __name__ == '__main__':
    epochs=[0,1,2,3]
    loss=[10,8,5,4]
    acc=[2,6,12,18]
    plot(epochs,loss,acc)