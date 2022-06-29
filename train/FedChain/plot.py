import os

import matplotlib.pyplot as plt
import numpy as np


def plot(x,y1,label1, dir="./results/", name="result"):
    plt.figure().clear()
    ax = plt.subplot(1,1,1)
    plt.sca(ax)
    plt.plot(x, y1, "r", label=label1)
    plt.legend()

    path = os.path.join(dir, name+".png")
    plt.savefig(path)

def compare(x,y1,label1, y2, label2, dir="./results/", name="result"):
    plt.figure().clear()
    # ax1 = plt.subplot(1, 2, 1)
    # plt.sca(ax1)
    # plt.plot(epochs,loss, "r", label="train_loss")
    # plt.legend()

    ax = plt.subplot(1,1,1)
    plt.sca(ax)
    plt.plot(x, y1, "r", label=label1)
    plt.plot(x, y2, "b", label=label2)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend()

    path = os.path.join(dir, name+".png")
    plt.savefig(path)

def save_array(x, y1, y2, dir="./results/", name="result"):
    list = np.vstack((x,y1,y2))

    path = os.path.join(dir, name + ".npy")
    np.save(path, list)


if __name__ == '__main__':
    # epochs=[0,1,2,3]
    # loss=[10,8,5,4]
    # acc=[2,6,12,18]
    #plot(epochs,acc, "acc", name="1")

    #save_array(epochs, acc, loss, name="1")

    # a = np.load("results/cifar100_result_iid_exchange/cifar100_result_iid_exchange.npy")
    # b = np.load("results/cifar100_result_iid_base/cifar100_result_iid_base.npy")

    a = np.load("results/cifar100_result_niid0.5_exchange/cifar100_result_niid0.5_exchange.npy")
    b = np.load("results/cifar100_result_niid0.5_base/cifar100_result_niid0.5_base.npy")

    # a = np.load("results/cifar10_result_iid1.0_exchange/cifar10_result_iid1.0_exchange.npy")
    # b = np.load("results/cifar10_result_iid1.0_base/cifar10_result_niid1.0_base.npy")

    # a = np.load("results/cifar10_result_niid0.1_exchange/cifar10_result_niid0.1_exchange.npy")
    # b = np.load("results/cifar10_result_niid0.1_base/cifar10_result_niid0.1_base.npy")

    #a = np.load("results/fashion_mnist_result_niid0.1_exchange/fashion_mnist_result_niid0.1_exchange.npy")
    #b = np.load("results/fashion_mnist_result_niid0.1_base/fashion_mnist_result_niid0.1_base.npy")

    # a = np.load("results/fashion_mnist_result_iid_exchange/fashion_mnist_result_iid_exchange.npy")
    # b = np.load("results/fashion_mnist_result_iid_base/fashion_mnist_result_iid_base.npy")

    compare(a[0], a[1], "fedchain_acc", b[1], "baseline_acc")