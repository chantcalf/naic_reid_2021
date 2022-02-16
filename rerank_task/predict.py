# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 19:12:47 2022

@author: chantcalf
"""
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

work_dir = os.path.realpath(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_dat(fpath):
    return np.fromfile(fpath, dtype='float32')


class TestDataSet:
    def __init__(self, dat_dir):
        names = sorted(os.listdir(dat_dir))
        self.fpaths = [os.path.join(dat_dir, i) for i in names]
        self.names = [i.replace('.dat', '.png') for i in names]
        self.num = len(self.names)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return index, read_dat(self.fpaths[index])


test_type = "B"


class DefaultCfg:
    query_path = os.path.join(work_dir, f'../data/test_{test_type}/query_feature_{test_type}')
    gallery_path = os.path.join(work_dir, f'../data/test_{test_type}/gallery_feature_{test_type}')
    query_batch_size = 20000
    gallery_batch_size = 10000
    num_workers = 0
    save_path = f"./sub_{test_type}.json"
    topk = 100


def get_norm(x):
    return x / torch.norm(x, dim=1, keepdim=True).clamp(min=1e-5)


def get_cosine_dist(x, y):
    x = get_norm(x)
    y = get_norm(y)
    return x @ y.transpose(0, 1)


def predict_res(cfg, query_loader, gallery_loader):
    res = []
    for q_id, q_data in tqdm(query_loader):
        start_time = time.time()
        q_data = q_data.to(device)
        best_dist = None
        best_id = None
        for g_id, g_data in tqdm(gallery_loader):
            g_data = g_data.to(device)
            g_id = g_id.to(device).unsqueeze(0).expand(q_data.shape[0], -1)
            dist = get_cosine_dist(q_data, g_data)
            if best_dist is None:
                best_dist = dist
                best_id = g_id
            else:
                best_dist = torch.cat([best_dist, dist], -1)
                best_id = torch.cat([best_id, g_id], -1)
            if best_dist.shape[1] > cfg.topk:
                best_dist, tid = torch.topk(best_dist, cfg.topk, dim=1)
                best_id = best_id.gather(1, tid)
        res.append([q_id, best_id, best_dist])
        print(f"cost {time.time() - start_time}s")
    return res


def main():
    cfg = DefaultCfg()
    query = TestDataSet(cfg.query_path)
    gallery = TestDataSet(cfg.gallery_path)
    query_loader = torch.utils.data.DataLoader(
        query, batch_size=cfg.query_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)
    gallery_loader = torch.utils.data.DataLoader(
        gallery, batch_size=cfg.gallery_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)
    print("start to predict")
    with torch.no_grad():
        res = predict_res(cfg, query_loader, gallery_loader)
    print("end predict")
    ans = dict()
    for i in range(len(res)):
        q_id = res[i][0]
        g_id = res[i][1].detach().cpu().numpy()
        b, n = g_id.shape
        for j in range(b):
            j_name = query.names[q_id[j]]
            ans[j_name] = [gallery.names[g_id[j][k]] for k in range(n)]

    with open(cfg.save_path, "w", encoding='utf8') as f:
        json.dump(ans, f)

    with open(cfg.save_path, "r", encoding='utf8') as f:
        content = f.read()

    with open(cfg.save_path, "w", encoding='utf8') as f:
        f.write(content.replace("png", "dat"))


if __name__ == "__main__":
    main()
