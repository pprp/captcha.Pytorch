from config.parameters import *
import torch as t
from torch import nn
import torch.nn.functional as F
import os
from torchvision import models
from torch.nn import init
import torch
from model.model import weights_init_kaiming, weights_init_classifier

class ClassBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 class_num,
                 dropout=False,
                 relu=False,
                 num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        #add_block += [nn.Linear(input_dim, num_bottleneck)]
        num_bottleneck = input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        f = self.add_block(x)
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(f)
        return x

class dense121(nn.Module):
    def __init__(self, class_num=62):
        super(dense121, self).__init__()
        self.model = models.densenet121(pretrained=True).features
        # .densenet121(pretrained=False).features
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.flat = torch.flatten(dims=1)
        # model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_of_feature = 1024
        self.classifier = ClassBlock(1024, 1024)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_of_feature, class_num)
        self.fc2 = nn.Linear(num_of_feature, class_num)
        self.fc3 = nn.Linear(num_of_feature, class_num)
        self.fc4 = nn.Linear(num_of_feature, class_num)

    def forward(self, x):
        x = self.model(x)
        x = self.relu(x)
        x = self.avgpool(x)
        # x = self.flat(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        x = self.drop(x)
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

    def load_model(self, weight_path):
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