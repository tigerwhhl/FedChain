import torch
import os
import gzip
import numpy as np
from torchvision import datasets, transforms

class GetDataSet(object):
    def __init__(self, dataset_name, is_iid):
        self.name = dataset_name
        self.isIID = is_iid #数据是否满足独立同分布
        self.train_data = None  # 训练集
        self.train_label = None  # 标签
        self.train_data_size = None  # 训练数据的大小
        self.test_data = None  # 测试数据集
        self.test_label = None  # 测试的标签
        self.test_data_size = None  # 测试集数据大小


        if self.name == "mnist":
            self.get_mnist_dataset()

        if self.name == "cifar":
            self.get_cifar_dataset()

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
        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)

        #test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
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


    def get_cifar_dataset(self):
        data_dir = "./data/CIFAR/"
        transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
            order = np.argsort(train_labels)
            distribution={}
            for label in order:
                    if train_labels[label] not in distribution:
                        distribution[train_labels[label]]=1
                    else:
                        distribution[train_labels[label]]+=1
            print("Label Distribution : {}".format(distribution))
            train_dataset.data = train_data[order]
            train_dataset.targets = train_labels[order]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

if __name__ == "__main__":
    mnist_dataset = GetDataSet("mnist", isIID=False)
    #cifar_dataset = GetDataSet("cifar", isIID=False)


