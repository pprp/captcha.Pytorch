# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:32:59 2019

@author: qmzhang
"""

import numpy as np
import pandas as pd


def model(testpath):
    # your model goes here
    # 在这里放入或者读入模型文件
    pass


    # the format of result-file
    # 这里可以生成结果文件
    ids = [str(x) + ".jpg" for x in range(1, 5001)]
    labels = ['test'] * 5000
    df = pd.DataFrame([ids, labels]).T
    df.columns = ['ID', 'label']
    return df
