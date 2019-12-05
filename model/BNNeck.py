from config.parameters import *
import torch as t
from torch import nn
import torch.nn.functional as F
import os
from torchvision import models
from torch.nn import init
import torch
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
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


class bnneck(nn.Module):
    def __init__(self, class_num=62):
        super(bnneck, self).__init__()
        resnet = ResNet(BasicBlock, [2, 2, 2, 2])
        self.base_model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.bnneck = nn.BatchNorm1d(256)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.reduce_layer = nn.Conv2d(512, 256, 1)

        # self.classifier = ClassBlock(512, 1024)
        self.fc1 = nn.Sequential(
            nn.Linear(256, class_num))
        self.fc2 = nn.Sequential(
            nn.Linear(256, class_num))
        self.fc3 = nn.Sequential(
            nn.Linear(256, class_num))
        self.fc4 = nn.Sequential(
            nn.Linear(256, class_num))

    def forward(self, x):
        bs = x.shape[0]
        x = self.base_model(x)
        x = self.maxpool(x)
        x = self.reduce_layer(x).view(bs, -1)
        feat = self.bnneck(x)
        if not self.training:
            feat = nn.functional.normalize(feat, dim=1, p=2)
        x1 = self.fc1(feat)
        x2 = self.fc2(feat)
        x3 = self.fc3(feat)
        x4 = self.fc4(feat)
        return x1, x2, x3, x4

    def save(self, circle):
        name = "./weights/bnneck" + str(circle) + ".pth"
        torch.save(self.state_dict(), name)
        name2 = "./weights/bnneck_new.pth"
        torch.save(self.state_dict(), name2)

    def load_model(self, weight_path):
        fileList = os.listdir("./weights/")
        # print(fileList)
        if "bnneck_new.pth" in fileList:
            name = "./weights/bnneck_new.pth"
            self.load_state_dict(t.load(name))
            print("the latest model has been load")
        elif os.path.isfile(weight_path):
            self.load_state_dict(t.load(weight_path))
            print("load %s success!" % weight_path)
        else:
            print("%s do not exists." % weight_path)
