import Augmentor
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as  T
from parameters import *
import torch as t

import re
from dataset import *
from torch.utils.data import DataLoader

def get_distortion_pipline(path, num):
    p = Augmentor.Pipeline(path)
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.sample(num)
    return p

def get_skew_tilt_pipline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.skew_tilt(probability=0.5,magnitude=0.02)
    p.skew_left_right(probability=0.5,magnitude=0.02)
    p.skew_top_bottom(probability=0.5, magnitude=0.02)
    p.skew_corner(probability=0.5, magnitude=0.02)
    p.sample(num)
    return p

def get_rotate_pipline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.rotate(probability=1,max_left_rotation=1,max_right_rotation=1)
    p.sample(num)
    return p

if __name__ == "__main__":
    times = 2
    path = r"C:\Users\pprp\Desktop\验证码\dataset5\train"
    num = len(os.listdir(path)) * times
    p = get_distortion_pipline(path, num)
    # p = get_rotate_pipline(path, num)
    # p.process()
    # augTrainDataset = augCaptcha("./data/auged_train", train=True)
    # trainDataset = Captcha("./data/train/", train=True)
    # testDataset = Captcha("./data/test/", train=False)
    # augTrainDataLoader = DataLoader(augTrainDataset, batch_size=1,
    #                                 shuffle=True, num_workers=4)
    # trainDataLoader = DataLoader(trainDataset, batch_size=1,
    #                              shuffle=True, num_workers=4)
    # testDataLoader = DataLoader(testDataset, batch_size=1,
    #                             shuffle=True, num_workers=4)
    
    # for data, label, data1, label1 in augTrainDataLoader,trainDataLoader:
    #     print(data.size(), label, data1.size(), label1)