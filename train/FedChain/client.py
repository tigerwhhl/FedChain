import models, torch, copy
import numpy as np
from datasets import GetDataSet
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class Client(object):

    def __init__(self, conf, train_dataset, id=-1):

        self.conf = conf

        model = models.get_model(conf["model_name"], load_from_local=True)
        # model = models.CNNMnist()
        if torch.cuda.is_available():
            model = model.cuda()
        self.local_model = model

        self.client_id = id

        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            all_range))


    def local_train(self, model):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # print(id(model))
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])
        # print(id(self.local_model))

        self.local_model.train()
        for e in range(self.conf["local_epochs"]):
            total_loss = 0.0

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                #print(target)
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                total_loss += loss.item()

                if self.conf['dp'] == True:
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0, norm_type=2)

                optimizer.step()
            print("Epoch {} done. Loss {}".format(e, total_loss))
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        # print(diff[name])

        return diff


class ClientGroup(object):
    def __init__(self, dataset_name, is_iid, num_of_clients):
        self.dataset_name = dataset_name
        self.is_iid = is_iid
        self.num_of_clients = num_of_clients
        self.client_set = {}

        self.test_data_loader=None

        self.dataset_allocation()

    def dataset_allocation(self):
        mnist_dataset = GetDataSet(dataSetName=self.dataset_name, isIID=self.is_iid)

        test_data = torch.tensor(mnist_dataset.test_data)
        test_label = torch.argmax(torch.tensor(mnist_dataset.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset(test_data,test_label), batch_size=32, shuffle=False)

        train_data = mnist_dataset.train_data
        train_label = mnist_dataset.train_label

        shard_size = mnist_dataset.train_data_size // self.num_of_clients // 2
        #print(shard_size) #3000
        shard_id = np.random.permutation(mnist_dataset.train_data_size // shard_size)
        #print(shared_id) #20

        for i in range(self.num_of_clients):
            shard_id1 = shard_id[i*2]
            shard_id2 = shard_id[i*2+1]

            data_shards1 = train_data[shard_id1 * shard_size: shard_id1 * shard_size + shard_size]
            data_shards2 = train_data[shard_id2 * shard_size: shard_id2 * shard_size + shard_size]
            label_shards1 = train_label[shard_id1 * shard_size: shard_id1 * shard_size + shard_size]
            label_shards2 = train_label[shard_id2 * shard_size: shard_id2 * shard_size + shard_size]

            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)
            #print(local_label)

            client = Client(conf=None, model=None, train_dataset= TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), id=i)
            break

if __name__ == "__main__":
    cient_group = ClientGroup(dataset_name="mnist",is_iid=False,num_of_clients=10)