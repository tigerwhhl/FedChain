import argparse, json
import datetime
import os
import logging

import numpy as np
import torch, random
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from server import *
from client import *
import datasets
import models
import plot

if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print(conf)

    dataset = datasets.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"])

    server = Server(conf, dataset.test_dataset, device=torch.device("cuda:0"))

    client = Client(
            conf=conf,
            train_dataset=dataset.train_dataset,
            test_dataset=dataset.test_dataset,
            id=0,
            global_model=server.global_model,
            device=torch.device("cuda:"+str(0)))
    print("Create 1 client for training in complete dataset.")

    client.local_train(client.local_model)