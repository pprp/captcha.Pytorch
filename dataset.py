import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as  T
from parameters import *
import torch as t

nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']


def StrtoLabel(Str):
    # print(Str)
    label = []
    for i in range(0, charNumber):
        if Str[i] >= '0' and Str[i] <= '9':
            label.append(ord(Str[i]) - ord('0'))
        elif Str[i] >= 'a' and Str[i] <= 'z':
            label.append(ord(Str[i]) - ord('a') + 10)
        else:
            label.append(ord(Str[i]) - ord('A') + 36)
    return label


def LabeltoStr(Label):
    Str = ""
    for i in Label:
        if i <= 9:
            Str += chr(ord('0') + i)
        elif i <= 35:
            Str += chr(ord('a') + i - 10)
        else:
            Str += chr(ord('A') + i - 36)
    return Str


class Captcha(data.Dataset):
    def __init__(self, root, train=True):
        self.imgsPath = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = T.Compose([
            T.Resize((ImageHeight, ImageWidth)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        imgPath = self.imgsPath[index]
        # print(imgPath)
        label = imgPath.split("/")[-1].split(".")[0]
        # print(label)
        labelTensor = t.Tensor(StrtoLabel(label))
        data = Image.open(imgPath)
        # print(data.size)
        data = self.transform(data)
        # print(data.shape)
        return data, labelTensor

    def __len__(self):
        return len(self.imgsPath)


if __name__ == '__main__':
    trainDataset = Captcha(trainRoot, train=True)
    # print(trainDataset.__getitem__(1000))
    # data = Image.open("./训练数据集\\ZzN3.jpg")
    # data.show()
    # trainDataset.__getitem__(4224)
    # labelTensor = t.zeros(tensorLength)
    labelTensor = StrtoLabel("34Tt")
    print(labelTensor)
    print(LabeltoStr(labelTensor))
