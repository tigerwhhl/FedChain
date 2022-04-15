import argparse, json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets
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

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []

    print("Create {} clients".format(conf["no_models"]))
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    print("Start Training...")
    epochs = []
    accuracy = []
    validloss = []
    for e in range(conf["global_epochs"]):

        candidates = random.sample(clients, conf["k"])

        weights = {}
        for c in candidates:
            weights[c]= random.uniform(0,1/conf["k"])

        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)


        for c in candidates:
            diff = c.local_train(server.global_model)
            print("client {} finish local train".format(c.client_id))

            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name] * weights[c])

        server.model_aggregate(weight_accumulator)

        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

        epochs.append(e)
        validloss.append(loss)
        accuracy.append(acc)

    plot.plot(epochs, validloss, accuracy)