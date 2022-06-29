import argparse, json
import torch, random
from server import *
from client import *
import datasets
import plot

#random.seed(42)

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

    best_acc = 0.0
    save_name = conf["dataset"] + "_result_" + ("iid" if conf["iid"] else "niid") + str(conf["niid_beta"]) + "_" + "onlyexhcange"
    save_dir = os.path.join("./results", save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_conf = json.dumps(conf)
    f2 = open(os.path.join(save_dir, "conf.json"), 'w')
    f2.write(save_conf)
    f2.close()

    records = {}
    for i in range(conf["client_num"]):
        records[i] = [0]*conf["client_num"]

    records[0][2] += 3
    print(records)

    dataset = datasets.GetDataSet(dataset_name=conf["dataset"], is_iid=conf["iid"], beta=conf["niid_beta"])

    server = Server(conf,dataset.test_dataset, device=torch.device("cuda:0"))

    clients = []
    print("Create {} clients".format(conf["client_num"]))

    #shard_size = dataset.train_data_size // conf["client_num"] // 2
    shard_size = dataset.train_data_size // conf["client_num"]
    shard_id = np.random.permutation(dataset.train_data_size // shard_size)
    train_data = dataset.train_dataset.data
    train_label = dataset.train_dataset.targets

    for i in range(conf["client_num"]):

        if conf["iid"] == False:
            subset = torch.utils.data.Subset(dataset.train_dataset, dataset.client_idcs[i])

        else:
            shards = list(range(shard_id[i] * shard_size, shard_id[i] * shard_size + shard_size))
            subset = torch.utils.data.Subset(dataset.train_dataset, shards)

        clients.append(Client(
            conf=conf,
            train_dataset=subset,
            test_dataset=dataset.test_dataset,
            id=i,
            global_model=server.global_model,
            device=torch.device("cuda:"+str(1))))

    #pool = ThreadPoolExecutor(max_workers=conf["k"])

    print("Start Training...")
    client_indices=list(range(conf["client_num"]))

    for e in range(conf["global_epochs"]):
        print("#######Global epoch {} start########".format(e))
        k = random.randint(3,conf["k"])
        candidates = random.sample(clients, k)

        if conf["exchange"]:

            for c in candidates:
                diff = c.local_train(c.local_model)

            p = random.random()
            if p <= conf["exchange_probability"]:
                print("Epoch {}: Start Exchanging the models by blockchain".format(e))

                dir = "./models/clients/client"
                exchange_indices = [random.sample(client_indices, random.randint(conf["exchange_min_num"], conf["exchange_max_num"])) for _ in candidates]
                print("exchange map: {}".format(exchange_indices))

                for i in range(len(candidates)):
                    print("The client {} load the model from clients {}".format(candidates[i].client_id , exchange_indices[i]))
                    for id in exchange_indices[i]:
                        records[candidates[i].client_id][id] += 1

                    acc1, _ = candidates[i].eval_model()
                    #backup = copy.deepcopy(candidates[i].local_model.state_dict())
                    print("Before : client {} valid acc {}".format(candidates[i].client_id, acc1))

                    candidates[i].save_model()
                    path = os.path.join("./models/clients/", 'client' + str(candidates[i].client_id), 'model.pth')

                    candidates[i].fuse_model_by_teachers([clients[id].local_model for id in exchange_indices[i]])

                    acc2, _ = candidates[i].eval_model()
                    print("After : client {} valid acc {}".format(candidates[i].client_id, acc2))


        if e % 10 == 0:
            accs = []
            for c in clients:
                acc, _ = c.eval_model()
                accs.append(acc)
            print("{} clients accuracy are: {}".format(conf["client_num"], accs))
            print(records)
            print("#######################################")