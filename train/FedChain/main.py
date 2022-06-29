import argparse, json
import datetime
import os
import logging

import numpy as np
import torch, random

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED, FIRST_COMPLETED

from server import *
from client import *
import datasets
import models
import plot

#random.seed(42)

if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if torch.cuda.is_available():
        print(torch.version.cuda)
        device0 = "cuda:0"
        device1 = "cuda:1"
        device2 = "cuda:2"
        device3 = "cuda:3"
    else:
        device0 = device1 = device2 = device3 = "cpu"

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print(conf)

    #executor = ThreadPoolExecutor(max_workers=conf['k'])

    session = requests.Session()
    org_server = "http://114.212.82.242:8080/"
    upload_local_api = "testupload"
    headers = {'content-type': 'application/json'}

    best_acc = 0.0
    save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else "niid") + str(conf["niid_beta"]) + "_" + (
        "exchange" if conf["exchange"] else "base")
    save_dir = os.path.join("./results", save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_conf = json.dumps(conf)
    f2 = open(os.path.join(save_dir, "conf.json"), 'w')
    f2.write(save_conf)
    f2.close()

    epochs = []
    accuracy = []
    validloss = []

    dataset = datasets.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"])
    global_small_subset = dataset.generate_global_small_dataset(beta=0.01)

    server = Server(conf,dataset.test_dataset, device=torch.device(device0))

    clients = []
    print("Create {} clients".format(conf["client_num"]))

    #shard_size = dataset.train_data_size // conf["client_num"] // 2
    shard_size = dataset.train_data_size // conf["client_num"]
    shard_id = np.random.permutation(dataset.train_data_size // shard_size)
    train_data = dataset.train_dataset.data
    train_label = dataset.train_dataset.targets

    for i in range(conf["client_num"]):
        # shard_id1 = shard_id[i * 2]
        # shard_id2 = shard_id[i * 2 + 1]
        #
        # shards1 = list(range(shard_id1 * shard_size, shard_id1 * shard_size + shard_size))
        # shards2 = list(range(shard_id2 * shard_size, shard_id2 * shard_size + shard_size))

        #shards = list(range(shard_id[i] * shard_size, shard_id[i] * shard_size + shard_size))

        #subset = torch.utils.data.Subset(dataset.train_dataset, shards1+shards2)

        if conf["iid"] == False:
            subset = torch.utils.data.Subset(dataset.train_dataset, dataset.client_idcs[i])

        else:
            shards = list(range(shard_id[i] * shard_size, shard_id[i] * shard_size + shard_size))
            subset = torch.utils.data.Subset(dataset.train_dataset, shards)

        clients.append(Client(
            conf=conf,
            #train_dataset=torch.utils.data.ConcatDataset([subset, global_subset]),
            train_dataset=subset,
            test_dataset=dataset.test_dataset,
            id=i,
            global_model=server.global_model,
            device=torch.device(device1)))

    print("Start Training...")
    client_indices=list(range(conf["client_num"]))

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

            if conf["model_store_in_file"]:
                for c in candidates:
                    c.load_model("./models/server/model.pth")
                    c.local_train(c.local_model)
                    if conf["communication_with_blockchain"]:
                        data = {
                            "org": "org"+ str(c.client_id),
                            "cur_hash_id": str(hash(frozenset(c.local_model.state_dict().values()))),
                            "model_url": "./models/clients/client" + str(c.client_id) + "/model.pth",
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time() * 1000)) / 1000))
                        }
                        response = session.post(org_server + upload_local_api, data=json.dumps(data), headers=headers)
                        print(response.content)

            else:
                for c in candidates:
                    c.local_train(server.global_model)

                #all_task = [executor.submit(c.local_train, server.global_model) for c in candidates]

            #wait(all_task, return_when=ALL_COMPLETED)

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
                    acc1, _ = candidates[i].eval_model()
                    #backup = copy.deepcopy(candidates[i].local_model.state_dict())
                    print("Before : client {} valid acc {}".format(candidates[i].client_id, acc1))

                    candidates[i].save_model()
                    path = os.path.join("./models/clients/", 'client' + str(candidates[i].client_id), 'model.pth')

                    #candidates[i].fuse_model_by_avg([clients[id].local_model for id in exchange_indices[i]])

                    # for id in exchange_indices[i]:
                    #     clients[id].fuse_model_by_distillation(candidates[i].local_model, candidates[i].client_id)
                    #     candidates[i].load_model(path)

                    candidates[i].fuse_model_by_teachers([clients[id].local_model for id in exchange_indices[i]])


                    acc2, _ = candidates[i].eval_model()
                    print("After : client {} valid acc {}".format(candidates[i].client_id, acc2))

                    # if acc2 < acc1 - 3.0:
                    #     print("client {} back up".format(candidates[i].client_id))
                    #     candidates[i].local_model.load_state_dict(backup)


                server.model_weight_aggregate([c.local_model for c in candidates])

            else:
                # for c in candidates:
                #     diff = c.local_train(server.global_model)

                #
                #     for name, params in server.global_model.state_dict().items():
                #         weight_accumulator[name].add_(diff[name] * weights[c])

                #server.model_update_aggregate(weight_accumulator)

                server.model_weight_aggregate([c.local_model for c in candidates])

        else:
            if conf["model_store_in_file"]:
                for c in candidates:
                    c.load_model("./models/server/model.pth")
                    c.local_train(c.local_model)

            else:
                for c in candidates:
                    c.local_train(server.global_model)


            #     for name, params in server.global_model.state_dict().items():
            #         weight_accumulator[name].add_(diff[name] * weights[c])

            server.model_weight_aggregate([c.local_model for c in candidates])

        acc, loss = server.model_eval()
        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

        if e > 50 and acc > best_acc:
            best_acc = acc
            torch.save(server.global_model.state_dict(), os.path.join(save_dir, 'global_model.pth'))

        epochs.append(e)
        validloss.append(loss)
        accuracy.append(acc)

        if e % 10 == 0 and e > 0:
            plot.plot(epochs, accuracy, label1="accuracy", dir=save_dir, name=save_name)
            plot.save_array(epochs, accuracy, validloss, dir=save_dir, name=save_name)

    print("Finish the federated learning, best acc: {}".format(best_acc))

    session.close()