from config.parameters import *
import torch as t
from torch import nn
import torch.nn.functional as F
import os
from torchvision import models
from torch.nn import init
import torch


class RES50(nn.Module):
    def __init__(self, class_num=62):
        super(RES50, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(512, 1024)
        self.fc1 = nn.Linear(1024, class_num)
        self.fc2 = nn.Linear(1024, class_num)
        self.fc3 = nn.Linear(1024, class_num)
        self.fc4 = nn.Linear(1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        return x1, x2, x3, x4

    def save(self, circle):
        name = "./weights/net" + str(circle) + ".pth"
        t.save(self.state_dict(), name)
        name2 = "./weights/net_new.pth"
        t.save(self.state_dict(), name2)

    def loadIfExist(self, weight_path):
        fileList = os.listdir("./weights/")
        # print(fileList)
        if "net_new.pth" in fileList:
            name = "./weights/net_new.pth"
            self.load_state_dict(t.load(name))
            print("the latest model has been load")
        elif os.path.isfile(weight_path):
            self.load_state_dict(t.load(weight_path))
            print("load %s success!" % weight_path)
        else:
            print("%s do not exists." % weight_path)