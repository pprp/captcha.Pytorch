from model import *
from dataset import *
from train import *
from parameters import *
import torch as t
from torch import nn
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as  T
from parameters import *
import torch as t
import csv


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
        label = imgPath.split("/")[-1]
        # print(label)
        
        # print(label)
        # labelTensor = t.Tensor(StrtoLabel(label))
        data = Image.open(imgPath)
        # print(data.size)
        data = self.transform(data)
        # print(data.shape)
        return data, label

    def __len__(self):
        return len(self.imgsPath)

def predict(model, dataLoader):
    f = open("./submission.csv","w")
    csv_writer = csv.writer(f)

    for circle, input in enumerate(dataLoader, 0):
        x, label = input
        label = list(label)[0]
        # print(label)
        if t.cuda.is_available():
            x = x.cuda()

        y1, y2, y3, y4 = model(x)
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
                         y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
        y = t.cat((y1, y2, y3, y4), dim=1)
        # print(x,label,y)
        decLabel = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
        print("predict %s is %s " % (label, decLabel))
        csv_writer.writerow([label,decLabel])
        # print("real: %s -> %s , %s" % (realLabel, decLabel, str(realLabel == decLabel)))
    f.close()

if __name__ == '__main__':
    model = ResNet(ResidualBlock)
    model.eval()
    model.loadIfExist()

    if t.cuda.is_available():
        model = model.cuda()
    userTestDataset = Captcha("./test/", train=True)
    userTestDataLoader = DataLoader(userTestDataset, batch_size=1,
                                    shuffle=False, num_workers=1)
    predict(model, userTestDataLoader)
