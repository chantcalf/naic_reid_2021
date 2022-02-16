# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:21:45 2022

@author: chantcalf
"""
import os
import numpy as np
from submit_compress import compress_all
from submit_reconstruct import reconstruct
from train_compress import read_dat

query_fea_dir = './query_feature'
save_dir = './reconstructed_query_feature'


def cal_loss(x, y):
    dif = x - y
    return np.sqrt(np.sum(dif * dif))


for byte in ["64", "128", "256"]:
    compress_all(query_fea_dir, int(byte))
    reconstruct(int(byte))
    loss = 0
    names = os.listdir(query_fea_dir)
    n = len(names)
    for name in names:
        x = read_dat(os.path.join(query_fea_dir, name))
        y = read_dat(os.path.join(save_dir, byte, name))
        loss += cal_loss(x, y)

    print(loss / n)