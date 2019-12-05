from model.model import *
from lib.dataset import *
from train import *
from config.parameters import *
import torch as t
from torch import nn
import torch.nn.functional as F
import os, shutil
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as  T
from config.parameters import *
import torch as t
import csv
import time

from model.dense import dense121
from model.senet import senet
from model.res18 import res18
from model.dualpooling import DualResNet
from model.BNNeck import bnneck

os.environ['CUDA_VISIBLE_DEVICES']='1'

class Dataset4Captcha(data.Dataset):
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
        # labelTensor = t.Tensor(StrtoLabel(label))
        data = Image.open(imgPath)
        # print(data.size)
        data = self.transform(data)
        # print(data.shape)
        return data, label

    def __len__(self):
        return len(self.imgsPath)

def predict(model, dataLoader, csv_file):
    f = open(csv_file,"w")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["ID","label"])
    print("\t%-9s\t%-4s" % ("ID", "label"))

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
        # print("predict %s is %s " % (label, decLabel))
        csv_writer.writerow([label,decLabel])
        print("%d\t%-9s\t%-4s" % (circle, label, decLabel))
        # print("real: %s -> %s , %s" % (realLabel, decLabel, str(realLabel == decLabel)))
    f.close()

def getLabel(model, x):
    y1, y2, y3, y4 = model(x)
    y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
        y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
    y = t.cat((y1, y2, y3, y4), dim=1)
    # print(x,label,y)
    decLabel = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
    return decLabel

def predict_all(model_list, dataLoader, csv_file):
    f = open(csv_file, "w")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["ID", "label"])
    print("\t%-9s\t%-4s" % ("ID", "label"))

    num_of_model = len(model_list)

    for circle, input in enumerate(dataLoader, 0):
        x, label = input
        label = list(label)[0]
        if t.cuda.is_available():
            x = x.cuda()
        
        result_list = []

        for i in range(num_of_model):
            model = model_list[i]
            decLabel = getLabel(model, x)
            result_list.append(decLabel)
        
        csv_writer.writerow([label, decLabel])
        # print("%d\t%-9s\t%-4s" % (circle, label, decLabel))
        # if result_list[0] != result_list[1] or \
        #    result_list[0] != result_list[2] or \
        #    result_list[1] != result_list[2]:
        if len(set(result_list)) != 1:
            print("%d\t%-9s\t" % (circle, label), end='')
            append_name = "_"
            for i in range(num_of_model):
                print(" %-5s" % result_list[i], end='')
                append_name = append_name + "_"+ str(i) + "_"+ result_list[i]
            print()
            shutil.copy(os.path.join("/home/sunqilin/dpj/captcha.Pytorch/test", label),
                        os.path.join("/home/sunqilin/dpj/captcha.Pytorch/wrong", 
                        label.split('.')[0]+append_name+".jpg"))
            time.sleep(1)

    f.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="weightpath")
    parser.add_argument("--weight_path", type=str,
                        default="./weights/bnnect_with_center_loss/bnneck_new.pth")
    parser.add_argument("--test_path", type=str, default="./test")
    args = parser.parse_args()

    model1 = ResNet(ResidualBlock)
    model2 = bnneck()
    model3 = DualResNet(ResidualBlock)
    model4 = bnneck()

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    model1.load_model("./weights/best/resNet_new.pth")
    model2.load_model("./weights/bnnect_with_center_loss/bnneck_new.pth")
    model3.load_model("./weights/DualresNet/DualresNet_new.pth")
    model4.load_model("./weights/bnnect_multistepLR/bnneck_new.pth")

    if t.cuda.is_available():
        model1 = model1.cuda()
        model2 = model2.cuda()
        model3 = model3.cuda()
        model4 = model4.cuda()

    userTestDataset = Dataset4Captcha(args.test_path, train=True)
    userTestDataLoader = DataLoader(userTestDataset, batch_size=1,
                                    shuffle=False, num_workers=1)

    model_list = []
    model_list.append(model1)
    model_list.append(model2)
    model_list.append(model3)
    model_list.append(model4)

    predict_all(model_list, userTestDataLoader, csv_file="./submission.csv")
    # predict(model, userTestDataLoader, csv_file="./submission.csv")
