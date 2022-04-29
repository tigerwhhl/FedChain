import argparse, json
import datetime
import os
import logging

import numpy as np
import torch, random

from server import *
from client import *
import datasets
import models
import plot

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print(conf)

    dataset = datasets.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"])
    server = Server(conf,dataset.test_dataset)

    clients = []
    print("Create {} clients".format(conf["client_num"]))

    shard_size = dataset.train_data_size // conf["client_num"] // 2
    shard_id = np.random.permutation(dataset.train_data_size // shard_size)
    train_data = dataset.train_dataset.data
    train_label = dataset.train_dataset.targets

    for i in range(conf["client_num"]):
        #clients.append(Client(conf, server.global_model, train_datasets, c))
        #clients.append(Client(conf, server.global_model, cifar_dataset.train_dataset, i))

        shard_id1 = shard_id[i * 2]
        shard_id2 = shard_id[i * 2 + 1]

        shards1 = list(range(shard_id1 * shard_size, shard_id1 * shard_size + shard_size))
        shards2 = list(range(shard_id2 * shard_size, shard_id2 * shard_size + shard_size))

        subset = torch.utils.data.Subset(dataset.train_dataset, shards1+shards2)
        clients.append(Client(conf=conf,
                        train_dataset=subset, test_dataset=dataset.test_dataset, id=i))

    print("Start Training...")
    client_indices=list(range(conf["client_num"]))
    epochs = []
    accuracy = []
    validloss = []
    for e in range(conf["global_epochs"]):

        k = random.randint(3,conf["k"])
        candidates = random.sample(clients, k)

        weights = {}
        for c in candidates:
            weights[c]= random.uniform(0,1/conf["k"])

        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)


        if conf["exchange"]:
            p = random.random()
            if p <= conf["exchange_probability"]:
                print("Epoch {}: Start Exchanging the models by blockchain".format(e))

                dir = "./models/clients/client"
                exchange_indices = [random.sample(client_indices, random.randint(conf["exchange_min_num"], conf["exchange_max_num"])) for _ in candidates]
                print("exchange map: {}".format(exchange_indices))

                # for i in range(conf["client_num"]):
                #     clients[i].save_model()

                for i in range(len(candidates)):
                    print("The client {} load the model from clients {}".format(candidates[i].client_id , exchange_indices[i]))
                    acc, _ = candidates[i].eval_model()
                    print("Before : client {} valid acc {}".format(candidates[i].client_id, acc))

                    candidates[i].save_model()
                    path = os.path.join("./models/clients/", 'client' + str(candidates[i].client_id), 'model.pth')

                    #candidates[i].fuse_model_by_avg([clients[id].local_model for id in exchange_indices[i]])

                    for id in exchange_indices[i]:
                        clients[id].fuse_model_by_distillation(candidates[i].local_model, candidates[i].client_id)
                        candidates[i].load_model(path)


                    acc, _ = candidates[i].eval_model()
                    print("After : client {} valid acc {}".format(candidates[i].client_id, acc))

                server.model_weight_aggregate([c.local_model for c in candidates])

            else:
                for c in candidates:
                    diff = c.local_train(server.global_model)
                    print("client {} finish local train".format(c.client_id))

                    for name, params in server.global_model.state_dict().items():
                        weight_accumulator[name].add_(diff[name] * weights[c])

                #server.model_update_aggregate(weight_accumulator)
                server.model_weight_aggregate([c.local_model for c in candidates])

        else:
            for c in candidates:
                diff = c.local_train(server.global_model)
                print("client {} finish local train".format(c.client_id))

                for name, params in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff[name] * weights[c])

            #server.model_update_aggregate(weight_accumulator)
            server.model_weight_aggregate([c.local_model for c in candidates])

        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

        epochs.append(e)
        validloss.append(loss)
        accuracy.append(acc)

        if e%10 ==0 and e>0:
            plot.plot(epochs, accuracy, label1="accuracy", name=conf["save_name"])
            plot.save_array(epochs, accuracy, validloss, name=conf["save_name"])