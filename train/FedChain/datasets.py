import random

import torch
import os
import gzip
import numpy as np
import torchvision.utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

np.random.seed(19)

class GetDataSet(object):
    def __init__(self, dataset_name, is_iid, beta=1.0):
        self.name = dataset_name
        self.isIID = is_iid #数据是否满足独立同分布
        self.beta = beta
        self.train_data = None  # 训练集
        self.train_label = None  # 标签
        self.train_data_size = None  # 训练数据的大小
        self.test_data = None  # 测试数据集
        self.test_label = None  # 测试的标签
        self.test_data_size = None  # 测试集数据大小


        if self.name == "mnist":
            self.get_mnist_dataset()

        if self.name == "fashion_mnist":
            self.get_fashion_mnist_dataset()

        if self.name == "cifar10":
            self.get_cifar10_dataset()

        if self.name == "cifar100":
            self.get_cifar100_dataset()

    def get_mnist_dataset(self):
        data_dir = "./data/"
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())

        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        #train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])

        #test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        test_dataset.data = torch.tensor(test_images)

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.data = torch.tensor(train_images[order])
            train_dataset.targets = torch.tensor(train_labels[order])
            #print(train_dataset.targets)

        else:
            order = np.argsort(train_labels)
            # print(order)
            #print(train_labels[order])
            distribution={}
            for label in order:
                    if train_labels[label] not in distribution:
                        distribution[train_labels[label]]=1
                    else:
                        distribution[train_labels[label]]+=1
            print("Label Distribution : {}".format(distribution))

            train_dataset.data = torch.tensor(train_images[order])
            train_dataset.targets = torch.tensor(train_labels[order])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        print("The shape of mnist data: {}".format(self.train_dataset.data.shape))
        print("The shape of mnist label: {}".format(self.train_dataset.targets.shape))

    def get_fashion_mnist_dataset(self, client_num=10, class_num=10):
        data_dir = "./data/"
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transforms.ToTensor())

        train_images = train_dataset.data.numpy()
        train_labels = train_dataset.targets.numpy()
        test_images = test_dataset.data.numpy()
        test_labels = test_dataset.targets.numpy()

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        # train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])

        # test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        test_dataset.data = torch.tensor(test_images)

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.data = torch.tensor(train_images[order])
            train_dataset.targets = torch.tensor(train_labels[order])
            # print(train_dataset.targets)

        else:
            # order = np.argsort(train_labels)
            # # print(order)
            # # print(train_labels[order])
            # distribution = {}
            # for label in order:
            #     if train_labels[label] not in distribution:
            #         distribution[train_labels[label]] = 1
            #     else:
            #         distribution[train_labels[label]] += 1
            # print("Label Distribution : {}".format(distribution))
            #
            # train_dataset.data = torch.tensor(train_images[order])
            # train_dataset.targets = torch.tensor(train_labels[order])

            train_dataset.data = torch.tensor(train_images)
            train_dataset.targets = torch.tensor(train_labels)

            label_distribution = np.random.dirichlet([self.beta] * client_num, class_num)
            class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs

            for idcs in client_idcs:
                a = train_labels[idcs]
                distribution = {}
                for label in a:
                    if label not in distribution:
                        distribution[label] = 1
                    else:
                        distribution[label] += 1
                print("Label Distribution : {}".format(distribution))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        print("The shape of fashion_mnist data: {}".format(self.train_dataset.data.shape))
        print("The shape of fashion_mnist label: {}".format(self.train_dataset.targets.shape))


    def get_cifar10_dataset(self, client_num=10, class_num=10):
        data_dir = "./data/CIFAR/"
        transform_train = transforms.Compose([
                    #transforms.RandomCrop(32, padding=4),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)

        train_data = train_dataset.data
        train_labels = np.array(train_dataset.targets)
        test_data = test_dataset.data
        test_labels = np.array(test_dataset.targets)
        test_dataset.targets = test_labels

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.data = train_data[order]
            train_dataset.targets = train_labels[order]

        else:
            #order = np.argsort(train_labels)
            # distribution={}
            # for label in order:
            #         if train_labels[label] not in distribution:
            #             distribution[train_labels[label]]=1
            #         else:
            #             distribution[train_labels[label]]+=1
            # print("Label Distribution : {}".format(distribution))
            #
            # iid_list = []
            # total_size = 5000
            # niid_size = int(total_size * self.beta)
            # iid_size = total_size - niid_size
            # for i in range(10):
            #     iid_list.extend(order[i*total_size+niid_size:(i+1)*total_size])
            # random.shuffle(iid_list)
            #
            # new_order = []
            # for i in range(10):
            #     new_order.extend(order[i*total_size:i*total_size+niid_size])
            #     new_order.extend(iid_list[i*iid_size:(i+1)*iid_size])
            #
            # train_dataset.data = train_data[new_order]
            # train_dataset.targets = train_labels[new_order]

            train_dataset.data = train_data
            train_dataset.targets = train_labels

            label_distribution = np.random.dirichlet([self.beta]*client_num, class_num)
            class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs


            for idcs in client_idcs:
                a=train_labels[idcs]
                distribution={}
                for label in a:
                        if label not in distribution:
                            distribution[label]=1
                        else:
                            distribution[label]+=1
                print("Label Distribution : {}".format(distribution))


        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print("The shape of cifar10 train data: {}".format(self.train_dataset.data.shape))
        print("The shape of cifar10 train label: {}".format(self.train_dataset.targets.shape))
        print("The shape of cifar10 test data: {}".format(self.test_dataset.data.shape))

    def get_cifar100_dataset(self, client_num=10, class_num=100):
        data_dir = "./data/CIFAR100/"
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                          transform=transform_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, transform=transform_test)

        train_data = train_dataset.data
        train_labels = np.array(train_dataset.targets)
        test_data = test_dataset.data
        test_labels = np.array(test_dataset.targets)
        test_dataset.targets = test_labels

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        if self.isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            train_dataset.data = train_data[order]
            train_dataset.targets = train_labels[order]

        else:
            # order = np.argsort(train_labels)
            #
            # distribution = {}
            # for label in order:
            #     if train_labels[label] not in distribution:
            #         distribution[train_labels[label]] = 1
            #     else:
            #         distribution[train_labels[label]] += 1
            # print("Label Distribution : {}".format(distribution))
            # train_dataset.data = train_data[order]
            # train_dataset.targets = train_labels[order]

            train_dataset.data = train_data
            train_dataset.targets = train_labels

            label_distribution = np.random.dirichlet([self.beta]*client_num, class_num)
            class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(class_num)]

            client_idcs = [[] for _ in range(client_num)]
            for c, fracs in zip(class_idcs, label_distribution):
                for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                    client_idcs[i] += [idcs]

            client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
            self.client_idcs = client_idcs


            for idcs in client_idcs:
                a=train_labels[idcs]
                distribution={}
                for label in a:
                        if label not in distribution:
                            distribution[label]=1
                        else:
                            distribution[label]+=1
                print("Label Distribution : {}".format(distribution))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        print("The shape of cifar100 train data: {}".format(self.train_dataset.data.shape))
        print("The shape of cifar100 train label: {}".format(self.train_dataset.targets.shape))
        print("The shape of cifar100 test data: {}".format(self.test_dataset.data.shape))

    def generate_global_small_dataset(self, beta=0.01, client_num=10):
        shard_size = self.train_data_size // client_num
        sample_size = int(shard_size * beta)
        global_ids = []
        for i in range(client_num):
            ids = random.sample(range(shard_size*i, shard_size*i + shard_size), sample_size)
            #print(ids)
            global_ids = global_ids + ids

        return global_ids

if __name__ == "__main__":
    #mnist_dataset = GetDataSet("fashion_mnist", is_iid=False)
    #mnist_dataset = GetDataSet("mnist", is_iid=False)
    #train_loader = torch.utils.data.DataLoader(mnist_dataset.test_dataset, batch_size=64)
    # for i, (images, labels) in enumerate(train_loader):
    #     if (i + 1) % 100 == 0:
    #         for j in range(len(images)):
    #             image = images[j].resize(28, 28)  # 将(1,28,28)->(28,28)
    #             plt.imshow(image)  # 显示图片,接受tensors, numpy arrays, numbers, dicts or lists
    #             plt.axis('off')  # 不显示坐标轴
    #             plt.title("$The {} picture in {} batch, label={}$".format(j + 1, i + 1, labels[j]))
    #             plt.show()

    cifar_dataset = GetDataSet("cifar10", is_iid=False, beta=0.3)
    print(cifar_dataset.train_dataset)
    # train_loader = torch.utils.data.DataLoader(cifar_dataset.train_dataset, batch_size=64)
    # for image,label in  train_loader:
    #     img = image[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
    #     img = img.numpy()  # FloatTensor转为ndarray
    #     img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
    #     # 显示图片
    #     plt.imshow(img)
    #     plt.show()