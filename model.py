from parameters import *
import torch as t
from torch import nn
import torch.nn.functional as F
import os

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=True)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.bottle = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=True)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=62):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=True),
            nn.ReLU(),
        )
        # https://blog.csdn.net/weixin_43624538/article/details/85049699
        # part 1: ResidualBlock basic
        # res18 2 2 2 2
        # res34 3 4 6 3
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(-1, 512)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4

    def save(self, circle):
        name = "./model/resNet" + str(circle) + ".pth"
        t.save(self.state_dict(), name)
        name2 = "./model/resNet_new.pth"
        t.save(self.state_dict(), name2)

    def loadIfExist(self, weight_path):
        fileList = os.listdir("./model/")
        # print(fileList)
        if "resNet_new.pth" in fileList:
            name = "./model/resNet_new.pth"
            self.load_state_dict(t.load(name))
            print("the latest model has been load")
        elif os.path.exists(weight_path):
            self.load_state_dict(t.load(weight_path))
            print("load %s success!" % weight_path)


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
        name = "./model/net" + str(circle) + ".pth"
        t.save(self.state_dict(), name)
        name2 = "./model/net_new.pth"
        t.save(self.state_dict(), name2)

    def loadIfExist(self, weight_path):
        fileList = os.listdir("./model/")
        # print(fileList)
        if "net_new.pth" in fileList:        
            name = "./model/net_new.pth"
            self.load_state_dict(t.load(name))
            print("the latest model has been load")
        else:
            self.load_state_dict(t.load(weight_path))
            print("load %s success!" % weight_path)
