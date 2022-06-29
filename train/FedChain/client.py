import os.path
import collections

import torch, copy
import numpy as np
from datasets import GetDataSet
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datasets
import models

import requests
import schedule
import time
import json
from concurrent.futures import ThreadPoolExecutor

np.random.seed(19)

class Client(object):

    def __init__(self, conf, train_dataset, test_dataset, id=-1, global_model=None, device=None):

        self.conf = conf

        #model = models.get_model(conf["model_name"], load_from_local=True)
        if self.conf["dataset"] == "cifar10":
            model = models.VGGCifar(num_classes=10)
        if self.conf["dataset"] == "cifar100":
            model = models.VGGCifar(num_classes=100)
        if self.conf["dataset"] == "mnist" or self.conf["dataset"] == "fashion_mnist":
            model = models.VGGMNIST()

        #if torch.cuda.is_available():
            #self.local_model = model.cuda()
        self.device = device
        self.local_model = model.to(self.device)

        if global_model:
            print("Initialize parameters from global model")
            for name, param in global_model.state_dict().items():
                self.local_model.state_dict()[name].copy_(param.clone())


        self.client_id = id

        self.round = -1

        self.train_dataset = train_dataset

        self.test_dataset = test_dataset

        all_range = list(range(len(self.train_dataset)))

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            all_range), num_workers=4)

        self.test_loader =  torch.utils.data.DataLoader(self.test_dataset, batch_size=self.conf["batch_size"],
                                                        shuffle=False, num_workers=4)

        map = {}
        for batch_id, batch in enumerate(self.train_loader):
            data, target = batch
            for t in target:
                if t.item() in map:
                    map[t.item()] += 1
                else:
                    map[t.item()] = 1
        print(map)

        if self.conf["communication_with_blockchain"]:
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.executor.submit(self.monitor_blockchain, 3)
            self.pull_global_model_ready = False


    def local_train(self, model, mode="train"):

        if self.conf["communication_with_blockchain"]:
            while not self.pull_global_model_ready:
                continue
            self.pull_global_model_ready = False
            print("Preparing for a new round of training")

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # print(id(model))
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])

        criterion = torch.nn.CrossEntropyLoss()

        if mode != "train":
            epoch = self.conf["exchange_local_epochs"]
        else:
            epoch = self.conf["local_epochs"]

        # acc, total_l = self.eval_model()
        # print("Valid accuracy {}".format(acc))

        self.local_model.train()

        for e in range(epoch):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                #print(target)
                #if torch.cuda.is_available():
                    #data = data.cuda()
                    #target = target.cuda()

                data = data.to(self.device)
                target = target.to(self.device, dtype=torch.long)

                optimizer.zero_grad()
                output = self.local_model(data)
                loss1 = criterion(output, target)

                proximal_term = 0.0
                for w, w_t in zip(self.local_model.parameters(), model.parameters()):
                    proximal_term += (w - w_t.clone().to(self.device)).norm(2)

                #print("{},{}".format(loss1, proximal_term))
                loss = loss1 + (0.1 / 2) * proximal_term

                loss.backward()
                #total_loss += loss.item()

                if self.conf['dp'] == True:
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0, norm_type=2)

                optimizer.step()

            acc, total_l = self.eval_model()
            print("Epoch {} done. Valid accuracy {}".format(e, acc))

        print("client {} finish local train".format(self.client_id))
        diff = dict()
        # for name, data in self.local_model.state_dict().items():
        #     diff[name] = (data - model.state_dict()[name].clone().to(self.device))
        # print(diff[name])

        torch.cuda.empty_cache()
        return diff

    def fuse_model_by_teachers(self, models):
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=5e-3, momentum=0.9,
                                    weight_decay=3e-4)

        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = torch.nn.KLDivLoss()

        #self.local_model.train()
        for model in models:
            model.eval()

        count = 0
        for e in range(self.conf["exchange_local_epochs"]):
            total_loss = 0.0
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                student_output = self.local_model(data)

                teacher_output = torch.zeros_like(student_output)
                weight = 1/len(models)
                for model in models:
                    teacher_output += weight * model(data)
                #print(teacher_output.data.max(1)[1])
                loss1 = criterion1(student_output, target)

                T = 1
                alpha = 1
                beta = 8
                loss2 = criterion2(F.log_softmax(student_output / T, dim=1),
                                   F.softmax(teacher_output / T, dim=1)) * T * T
                loss = alpha * loss1 + beta * loss2

                if count < 0:
                    print("{}, {}".format(loss1, loss2))
                    count += 1

                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            #acc, total_l = self.eval_model()
            print("Epoch {} done. Loss {}".format(e, total_loss))



    def fuse_model_by_distillation(self, model, id):
        optimizer = torch.optim.SGD(model.parameters(), lr=4e-3, momentum=0.9,
                                    weight_decay=3e-4)

        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = torch.nn.KLDivLoss()

        self.local_model.eval()
        model.train()

        for e in range(self.conf["exchange_local_epochs"]):
            total_loss = 0.0
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                data = data.to(self.device)
                target = target.to(self.device, dtype=torch.long)

                optimizer.zero_grad()
                teacher_output = self.local_model(data)
                student_output = model(data)

                loss1 = criterion1(student_output, target)

                T = 1
                alpha = 1
                beta = 10
                loss2 = criterion2(F.log_softmax(student_output/T, dim=-1), F.softmax(teacher_output/T, dim=-1))*T*T
                loss = alpha*loss1 + beta*loss2

                #loss = loss1

                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            print("Epoch {} done. Loss {}".format(e, total_loss))

        torch.save(model.state_dict(), os.path.join('./models/clients/', 'client'+str(id), 'model.pth'))


    def fuse_model_by_avg(self, models):
        agg_state_dict = collections.OrderedDict()

        for key, param in self.local_model.state_dict().items():
            sum = torch.zeros_like(param)
            for model in models:
                 sum.add_(model.state_dict()[key])

            sum.add_(param)
            sum = torch.div(sum, len(models)+1)
            agg_state_dict[key] = sum

        self.local_model.load_state_dict(agg_state_dict)

    def save_model(self, dir='./models/clients/', client_id=-1):
        if client_id!=-1:
            path = os.path.join(dir, 'client' + str(client_id), 'model.pth')
        else:
            path = os.path.join(dir, 'client'+str(self.client_id), 'model.pth')
        print("Save model to {}".format(path))
        torch.save(self.local_model.state_dict(), path)

    def load_model(self, path):
        print("Load model from {}".format(path))
        self.local_model.load_state_dict(torch.load(path))

    def eval_model(self):
        self.local_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        count = 0
        for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            dataset_size += data.size()[0]

            data = data.to(self.device)
            target = target.to(self.device, dtype=torch.long)

            output = self.local_model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability

            if count < 0:
                print(pred)
                count += 1

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        #print("Valid dataset acc: {}, total_l: {}".format(acc, total_l))
        return acc, total_l

    def save_model(self, dir='./models/clients/', client_id=-1):
        if client_id!=-1:
            path = os.path.join(dir, 'client' + str(client_id), 'model.pth')
        else:
            path = os.path.join(dir, 'client'+str(self.client_id), 'model.pth')
        print("Save model to {}".format(path))
        torch.save(self.local_model.state_dict(), path)


    def monitor_blockchain(self,interval):
        session = requests.Session()
        org_server = "http://114.212.82.242:8080/"
        monitor_api = "test"

        def monitor_block():
            res = session.get(org_server + monitor_api).content
            block = json.loads(res) #dict
            #print(block)

            if block["round"] > self.round:
                self.pull_global_model_ready = True
                self.round = block["round"]


        schedule.every(interval).seconds.do(monitor_block)
        while True:
            schedule.run_pending()
            time.sleep(1)

class ClientGroup(object):
    def __init__(self, conf):
        self.conf = conf
        self.test_data_loader=None
        self.clients= []
        self.dataset_allocation()

    def dataset_allocation(self):
        dataset = datasets.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"])

        self.test_data_loader = torch.utils.data.DataLoader(dataset.test_dataset, batch_size=32, shuffle=False, num_workers=4)

        shard_size = dataset.train_data_size // conf["client_num"] // 2
        shard_id = np.random.permutation(dataset.train_data_size // shard_size)
        train_data = dataset.train_dataset.data
        train_label = dataset.train_dataset.targets

        for i in range(conf["client_num"]):
            # clients.append(Client(conf, server.global_model, train_datasets, c))
            # clients.append(Client(conf, server.global_model, cifar_dataset.train_dataset, i))

            #shard_id1 = shard_id[i * 2]
            #shard_id2 = shard_id[i * 2 + 1]
            #shards1 = list(range(shard_id1 * shard_size, shard_id1 * shard_size + shard_size))
            #shards2 = list(range(shard_id2 * shard_size, shard_id2 * shard_size + shard_size))

            subset = torch.utils.data.Subset(dataset.train_dataset, dataset.client_idcs[i])
            self.clients.append(Client(conf=conf, train_dataset=subset, test_dataset=dataset.test_dataset, id=i, device=torch.device("cuda:"+str(0))))

if __name__ == "__main__":
    with open("./utils/test-conf.json", 'r') as f:
        conf = json.load(f)
    client_group = ClientGroup(conf=conf)

    client0 = client_group.clients[0]
    client1 = client_group.clients[1]
    client2 = client_group.clients[2]
    client3 = client_group.clients[3]


    client0.local_train(client0.local_model)
    client1.local_train(client1.local_model)
    #client2.local_train(client2.local_model)
    #client3.local_train(client3.local_model)

    client0.fuse_model_by_teachers([client1.local_model])
    acc, _ = client0.eval_model()
    print(acc)

    #client2.fuse_model_by_distillation(client1.local_model, 1)
    # client1.load_model(os.path.join("./models/clients/", 'client1', 'model.pth'))
    # acc, _ = client1.eval_model()
    # print(acc)
    #
    # client3.fuse_model_by_distillation(client1.local_model, 1)
    # client1.load_model(os.path.join("./models/clients/", 'client1', 'model.pth'))
    # acc, _ = client1.eval_model()
    # print(acc)
    #
    # client4.fuse_model_by_distillation(client1.local_model, 1)
    # client1.load_model(os.path.join("./models/clients/", 'client1', 'model.pth'))
    # acc, _ = client1.eval_model()
    # print(acc)

    # client2.local_train(client1.local_model)
    # client3.local_train(client1.local_model)
    # client4.local_train(client1.local_model)
