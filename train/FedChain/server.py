import collections
import os
import models, torch
import requests
import schedule
import time
import json
from concurrent.futures import ThreadPoolExecutor

class Server(object):

    def __init__(self, conf, eval_dataset, device=None):

        self.conf = conf

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True, num_workers=4)

        #model = models.get_model(conf["model_name"], load_from_local=True)
        # model = models.CNNMnist()
        if self.conf["dataset"] == "cifar10":
            model = models.VGGCifar(num_classes=10)

        if self.conf["dataset"] == "cifar100":
            model = models.VGGCifar(num_classes=100)

        if self.conf["dataset"] == "mnist" or self.conf["dataset"] == "fashion_mnist":
            model = models.VGGMNIST()

        self.device = device
        self.global_model = model.to(self.device)
        #self.global_model = model

        if self.conf["model_store_in_file"]:
            self.save_model()

        if self.conf["communication_with_blockchain"]:
            self.session = requests.Session()
            self.org_server = "http://114.212.82.242:8080/"
            self.update_global_api = "updateGlobal"

            self.executor = ThreadPoolExecutor(max_workers=1)
            self.executor.submit(self.monitor_blockchain, 5)
            self.fedavg_ready = False


    def model_update_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            #update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            update_per_layer = weight_accumulator[name]
            if self.conf['dp']:
                sigma = self.conf['sigma']
                #为梯度添加高斯噪声
                if torch.cuda.is_available():
                    noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                else:
                    noise = torch.FloatTensor(update_per_layer.shape).normal_(0, sigma)
                #print(update_per_layer)
                update_per_layer.add_(noise)

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_weight_aggregate(self, models):
         if self.conf["communication_with_blockchain"]:
            while not self.fedavg_ready:
                continue
            self.fedavg_ready = False
            print("Having enough local updates")

         fed_state_dict = collections.OrderedDict()

         for key, param in self.global_model.state_dict().items():
            sum = torch.zeros_like(param)
            for model in models:
                 sum.add_(model.state_dict()[key].clone().to(self.device))
            sum = torch.div(sum, len(models))
            fed_state_dict[key] = sum

         self.global_model.load_state_dict(fed_state_dict)

         if self.conf["model_store_in_file"]:
             self.save_model()

         if self.conf["communication_with_blockchain"]:
            data = {
                "cur_hash_id": str(hash(frozenset(self.global_model.state_dict().values()))),
                "cur_model_url": "./models/server/model.pth",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time() * 1000)) / 1000))
            }
            response = self.session.post(self.org_server + self.update_global_api, data=json.dumps(data), headers={'content-type': 'application/json'})
            print(response.content)


    def model_eval(self):
        self.global_model.eval()

        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            data = data.to(self.device)
            target = target.to(self.device, dtype=torch.long)

            output = self.global_model(data)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l

    def save_model(self, dir='./models/server/', client_id=-1):
        path = os.path.join(dir, 'model.pth')
        print("Save global model to {}".format(path))
        torch.save(self.global_model.state_dict(), path)

    def monitor_blockchain(self,interval):
        monitor_api = "test"

        def monitor_block():
            res = self.session.get(self.org_server + monitor_api).content
            block = json.loads(res) #dict
            #print(block)

            if block["uploadCount"] >= block["triggerAvgNum"]:
                self.fedavg_ready = True


        #schedule.clear()
        schedule.every(interval).seconds.do(monitor_block)
        while True:
            schedule.run_pending()
            time.sleep(1)
