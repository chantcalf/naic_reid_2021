# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:41:54 2022

@author: chantcalf
"""
import math
import os
import random
import time
from functools import partial

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from config import Logger, LOG_DIR, TRAIN_DATA_DIR, NUM_WORKERS
from sub_models import TrainModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOGGER = Logger(os.path.join(LOG_DIR, "train.txt")).logger


def read_dat(fpath):
    return np.fromfile(fpath, dtype='float32')


def get_non_zero(ds):
    res = 0
    n = 0
    for data in ds:
        res = (res * n + data.abs().sum(0)) / (n + 1)
        n += 1
    return [i for i in range(2048) if res[i] > 1e-5]


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class DataSet:
    def __init__(self, fpaths):
        self.fpaths = fpaths
        self.num = len(self.fpaths)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return read_dat(self.fpaths[index])


def set_seed(seed):
    os.environ["PYTHONASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate(model, best_loss, test_loader, test_ds, cfg):
        model.eval()
        tloss = [0, 0, 0]
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(device)
                for j, bit in enumerate(cfg.bs):
                    y = model.encoder(x, bit)
                    y = model.decoder(y, bit)
                    tloss[j] = tloss[j] + (y - x).pow(2).sum(-1).sqrt().sum().item()
            tloss = [i / len(test_ds) for i in tloss]
        average_loss = sum(tloss)
        lossp = ", ".join([f"{i:.3f}" for i in tloss])
        LOGGER.info(f"average_loss={average_loss:.3f}, best_loss={best_loss:.3f}, " + lossp)
        if average_loss < best_loss:
            torch.save(model.state_dict(), cfg.model_path)
            LOGGER.info("Model saved")
            best_loss = average_loss
        return best_loss


def lr_scheduler(step, warm_up_step, max_step):
    if step < warm_up_step:
        return 1e-2 + (1 - 1e-2) * step / warm_up_step
    return 0.1 + (1 - 0.1) * 0.5 * (1 + math.cos((step - warm_up_step) / (max_step - warm_up_step) * math.pi))


class DefaultCfg:
    all_data = [
        os.path.join(TRAIN_DATA_DIR, 'train/train_feature'),
        os.path.join(TRAIN_DATA_DIR, 'test_A/query_feature_A'),
        os.path.join(TRAIN_DATA_DIR, 'test_A/gallery_feature_A')
    ]
    seed = 1992
    batch_size = 256
    epochs = 200
    warmup = 1000
    learning_rate = 1e-3
    weight_decay = 0.05
    train_sp = 0.9
    num_workers = NUM_WORKERS
    bs = [64, 128, 256]
    model_path = './compress.pth'
    ema_decay = 0.999


def main():
    LOGGER.info("start")
    cfg = DefaultCfg()
    set_seed(cfg.seed)
    fpaths = []
    for fdir in cfg.all_data:
        names = sorted(os.listdir(fdir))
        fpaths = fpaths + [os.path.join(fdir, name) for name in names]
    n = len(fpaths)
    ds = DataSet(fpaths)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)
    select_index = get_non_zero(dl)
    LOGGER.info(f"select index {len(select_index)}")
    del ds, dl
    all_ids = list(range(len(fpaths)))
    random.shuffle(all_ids)
    train_n = int(n * cfg.train_sp)
    fpaths = np.array(fpaths)
    train_ds = DataSet(fpaths[all_ids[:train_n]])
    test_ds = DataSet(fpaths[all_ids[train_n:]])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    model = TrainModel(cfg.bs, select_index)
    model.to(device)
    ema = EMA(model, cfg.ema_decay)
    ema.register()
    steps_per_epoch = len(train_ds) // cfg.batch_size
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    max_step = steps_per_epoch * cfg.epochs
    scheduler = LambdaLR(optimizer=optimizer,
                         lr_lambda=partial(lr_scheduler,
                                           warm_up_step=cfg.warmup,
                                           max_step=max_step))
    
    best_loss = np.inf
    for epoch in range(cfg.epochs):
        LOGGER.info("###########")
        LOGGER.info(f"epoch {epoch}")
        epoch_start_time = time.time()
        model.train()
        for i, x in enumerate(train_loader):
            x = x.to(device)
            losses = model(x)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            scheduler.step()
            ema.update()
            if i % 100 == 0:
                loss_print = "\t".join([f"{key} {losses[key].item():.3f}" for key in losses])
                LOGGER.info(f'Epoch: [{epoch}] step: [{i}/{steps_per_epoch}]\t' + loss_print)

        best_loss = validate(model, best_loss, test_loader, test_ds, cfg)
        ema.apply_shadow()
        best_loss = validate(model, best_loss, test_loader, test_ds, cfg)
        ema.restore()

        LOGGER.info(f"cost {time.time() - epoch_start_time}s")
        LOGGER.info("###########")


if __name__ == "__main__":
    main()
