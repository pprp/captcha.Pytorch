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

class res18(nn.Module):
    def __init__(self, class_num=62):
        super(res18, self).__init__()
        model_ft = ResNet(BasicBlock, [2, 2, 2, 2])
        self.base_model = nn.Sequential(*list(model_ft.children())[:-3])
        # attention schema
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sign = nn.Sigmoid()
        in_plances = 256
        ratio = 8
        self.a_fc1 = nn.Conv2d(in_plances,in_plances//ratio,1,bias=False)
        self.a_relu = nn.ReLU()
        self.a_fc2 = nn.Conv2d(in_plances//ratio, in_plances, 1, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.reduce_layer = nn.Conv2d(512, 256, 1)

        # self.classifier = ClassBlock(512, 1024)
        self.fc1 = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(256, class_num))
        self.fc2 = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(256, class_num))
        self.fc3 = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(256, class_num))
        self.fc4 = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(256, class_num))

    def forward(self, x):
        bs = x.shape[0]
        x = self.base_model(x)
        # channel attention   
        avgout = self.a_fc2(self.a_relu(self.a_fc1(self.avgpool(x))))
        maxout = self.a_fc2(self.a_relu(self.a_fc1(self.maxpool(x))))
        ca = self.sign(avgout+maxout)
        # joint
        x = x * ca.expand_as(x)

        # fuse avgpool and maxpool
        xx1 = self.avg_pool(x)#.view(bs, -1).squeeze()
        xx2 = self.max_pool(x)#.view(bs, -1).squeeze()
        # xx1 = self.avg_pool(x)
        # xx2 = self.max_pool(x)
        # fuse the feature by concat
        x = torch.cat([xx1, xx2], dim=1)
        x = self.reduce_layer(x).view(bs,-1)
        # print(x.shape)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        return x1, x2, x3, x4

    def save(self, circle):
        name = "./weights/res18" + str(circle) + ".pth"
        torch.save(self.state_dict(), name)
        name2 = "./weights/res18_new.pth"
        torch.save(self.state_dict(), name2)

    def load_model(self, weight_path):
        fileList = os.listdir("./weights/")
        # print(fileList)
        if "res18_new.pth" in fileList:
            name = "./weights/res18_new.pth"
            self.load_state_dict(t.load(name))
            print("the latest model has been load")
        elif os.path.isfile(weight_path):
            self.load_state_dict(t.load(weight_path))
            print("load %s success!" % weight_path)
        else:
            print("%s do not exists." % weight_path)
