import Augmentor
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as  T
from parameters import *
import torch as t
# from dataAug import get_aug_pipline
import re
from dataset import *
from torch.utils.data import DataLoader



# def get_aug_pipline():
#     p = Augmentor.Pipeline("./data/train")
#     p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
#     p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
#     # p.flip_left_right(probability=0.5)
#     # p.flip_top_bottom(probability=0.5)
#     p.sample(7976)
#     return p


if __name__ == "__main__":
    # p = get_aug_pipline()
    # p.process()
    augTrainDataset = augCaptcha("./data/auged_train", train=True)
    trainDataset = Captcha("./data/train/", train=True)
    testDataset = Captcha("./data/test/", train=False)
    augTrainDataLoader = DataLoader(augTrainDataset, batch_size=1,
                                    shuffle=True, num_workers=4)
    trainDataLoader = DataLoader(trainDataset, batch_size=1,
                                 shuffle=True, num_workers=4)
    testDataLoader = DataLoader(testDataset, batch_size=1,
                                shuffle=True, num_workers=4)
    
    for data, label, data1, label1 in augTrainDataLoader,trainDataLoader:
        print(data.size(), label, data1.size(), label1)