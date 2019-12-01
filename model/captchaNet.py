from config.parameters import *
import torch as t
from torch import nn
import torch.nn.functional as F
import os
from torchvision import models
from torch.nn import init
import torch


class CaptchaNet(nn.Module):
    def __init__(self):
        super(CaptchaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.conv3 = nn.Conv2d(10, 16, 6)
        self.fc1 = nn.Linear(4 * 12 * 16, 512)
        # 这是四个用于输出四个字符的线性层
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc41 = nn.Linear(128, 62)
        self.fc42 = nn.Linear(128, 62)
        self.fc43 = nn.Linear(128, 62)
        self.fc44 = nn.Linear(128, 62)

    def forward(self, x):
        # 输入为3*128*64，经过第一层为5*62*30
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 输出形状10*29*13
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # 输出形状16*12*4
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # print(x.size())
        x = x.view(-1, 768)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x1 = F.softmax(self.fc41(x), dim=1)
        x2 = F.softmax(self.fc42(x), dim=1)
        x3 = F.softmax(self.fc43(x), dim=1)
        x4 = F.softmax(self.fc44(x), dim=1)
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
        else:
            self.load_state_dict(t.load(weight_path))
            print("load %s success!" % weight_path)

