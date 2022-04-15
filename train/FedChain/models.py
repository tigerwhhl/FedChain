import torch
import math
import os
import glob
from torchvision import models

def get_model(name="vgg16",  model_dir="./models/checkpoints/", pretrained=True, load_from_local=False):

    if load_from_local:
        print("Load model from local dir {}".format(model_dir))
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

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model

def model_norm(model_1, model_2):
    squared_sum = 0
    for name, layer in model_1.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
    return math.sqrt(squared_sum)


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = './models/'
    get_model(name="alexnet", load_from_local=False)
