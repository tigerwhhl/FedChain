def load_data(self, isIID):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                             transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
    train_data = train_set.data  # (50000, 32, 32, 3)
    train_labels = train_set.targets
    train_labels = np.array(train_labels)  # 将标签转化为
    print(type(train_labels))  # <class 'numpy.ndarray'>
    print(train_labels.shape)  # (50000,)

    test_data = test_set.data  # 测试数据
    test_labels = test_set.targets
    test_labels = np.array(test_labels)
    # print()

    self.train_data_size = train_data.shape[0]
    self.test_data_size = test_data.shape[0]

    # 将训练集转化为（50000，32*32*3）矩阵
    train_images = train_data.reshape(train_data.shape[0],
                                      train_data.shape[1] * train_data.shape[2] * train_data.shape[3])
    print(train_images.shape)
    # 将测试集转化为（10000，32*32*3）矩阵
    test_images = test_data.reshape(test_data.shape[0],
                                    test_data.shape[1] * test_data.shape[2] * test_data.shape[3])

    # ---------------------------归一化处理------------------------------#
    train_images = train_images.astype(np.float32)
    # 数组对应元素位置相乘
    train_images = np.multiply(train_images, 1.0 / 255.0)
    # print(train_images[0:10,5:10])
    test_images = test_images.astype(np.float32)
    test_images = np.multiply(test_images, 1.0 / 255.0)
    # ----------------------------------------------------------------#

    '''
        一工有60000个样本
        100个客户端
        IID：
            我们首先将数据集打乱，然后为每个Client分配600个样本。
        Non-IID：
            我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，
            然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
    '''
    if isIID:
        # 这里将50000 个训练集随机打乱
        order = np.arange(self.train_data_size)
        np.random.shuffle(order)
        self.train_data = train_images[order]
        self.train_label = train_labels[order]
    else:
        # 按照标签的
        # labels = np.argmax(train_labels, axis=1)
        # 对数据标签进行排序
        order = np.argsort(train_labels)
        print("标签下标排序")
        print(train_labels[order[20000:25000]])
        self.train_data = train_images[order]
        self.train_label = train_labels[order]

    self.test_data = test_images
    self.test_label = test_labels