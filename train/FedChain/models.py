import torch
import math
import os
import glob
from torchvision import models
from torch import nn
import torch.nn.functional as F

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class VGGMNIST(nn.Module):
    def __init__(self):
        super(VGGMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)

class VGGCifar(nn.Module):
    def __init__(self, num_classes=10, model_name="vgg16"):
        super(VGGCifar, self).__init__()
        net = get_model(model_name, load_from_local=True)
        net.classifier = nn.Sequential()

        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(name="vgg16",  model_dir="./models/checkpoints/", pretrained=True, load_from_local=False):

    if load_from_local:
        #print("Load model from local dir {}".format(model_dir))
        model = eval('models.%s(pretrained=False)' % name)
        path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % name)
        model_path = glob.glob(path_format)[0]
        model.load_state_dict(torch.load(model_path))

    else:
        print("Download model from Internet")
        if name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        elif name == "densenet121":
            model = models.densenet121(pretrained=pretrained)
        elif name == "alexnet":
            model = models.alexnet(pretrained=pretrained)
        elif name == "vgg16":
            model = models.vgg16(pretrained=pretrained)
        elif name == "vgg19":
            model = models.vgg19(pretrained=pretrained)
        elif name == "inception_v3":
            model = models.inception_v3(pretrained=pretrained)
        elif name == "googlenet":
            model = models.googlenet(pretrained=pretrained)

    return model
    # if torch.cuda.is_available():
    #     return model.cuda()
    # else:
    #     return model


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = './models/'
    get_model(name="alexnet", load_from_local=False)
