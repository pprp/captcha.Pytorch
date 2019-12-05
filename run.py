#! /usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os

import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils import data
from torchvision import transforms as T

from config.parameters import *
from lib.dataset import *
from model.BNNeck import bnneck
from model.dense import dense121
from model.dualpooling import DualResNet
from model.model import *
from model.res18 import res18
from model.senet import senet
from predict import Dataset4Captcha, predict, predict_all
from train import *

if __name__ == "__main__":

    #################### 不可修改区域开始 ######################
    # testpath = '/home/data/'						#测试集路径。包含验证码图片文件
    # result_folder_path = '/code/result/submission.csv'	#结果输出文件路径
    testpath = "./test"
    result_folder_path = "./submissionaaa.csv"
	#################### 不可修改区域结束 ######################
    # testpath = './test/'						#测试集路径。包含验证码图片文件
    # result_folder_path = './result/submission.csv'	#结果输出文件路径
    print("reading start!")
    # pic_names = [str(x) + ".jpg" for x in range(1, 5001)]
    # pics = [imageio.imread(testpath + pic_name) for pic_name in pic_names]
    print("reading end!")
	### 调用自己的工程文件，并这里生成结果文件（dataframe）
    # result = testmodel.model(testpath)
    # print(result)
	# # 注意路径不能更改，index需要设置为None
    # result.to_csv(result_folder_path, index=None)
	# ### 参考代码结束：输出标准结果文件
    weight_path = r"./resNet_new.pth"
    model = ResNet(ResidualBlock)
    model.eval()
    model.load_model(weight_path)

    if t.cuda.is_available():
        model = model.cuda()

    tdataset = Dataset4Captcha(testpath, train=False)
    tdataloader = DataLoader(tdataset, batch_size=1, num_workers=1, shuffle=False)
    predict(model, tdataloader, csv_file=result_folder_path)
