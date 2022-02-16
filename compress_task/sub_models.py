# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:06:45 2022

@author: chantcalf
"""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VQ_B = 8
VQ_DIM = 8


class VQ(nn.Module):
    def __init__(self, b, dim):
        super().__init__()
        k = 2 ** b
        self.k = k
        self.dim = dim
        embed = torch.randn(k, dim)
        self.embed = nn.Parameter(embed)

    @torch.no_grad()
    def quant(self, x):
        dist = x.unsqueeze(-2) - self.embed.unsqueeze(0).unsqueeze(1)
        _, ind = dist.pow(2).sum(-1).min(-1)
        return ind

    @torch.no_grad()
    def dequant(self, x):
        return F.embedding(x, self.embed)

    def forward(self, x):
        # x: (b, n, dim)
        dist = x.pow(2).sum(-1).unsqueeze(-1)
        dist = dist + self.embed.pow(2).sum(-1).unsqueeze(0).unsqueeze(1)
        dist = dist - x @ self.embed.transpose(0, 1) * 2
        _, ind = dist.min(-1)  # (b, n)
        qx = F.embedding(ind, self.embed)  # (b, n, dim)

        loss1 = self.dist(qx, x.detach()).mean()
        loss2 = self.dist(qx.detach(), x).mean()
        return (qx - x).detach() + x, loss1, loss2

    @staticmethod
    def dist(x, y):
        return (x - y).pow(2).mean(-1)


class MLP(nn.Module):
    def __init__(self, indim, hidden, act=nn.GELU):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(indim, hidden),
            act(),
            nn.Linear(hidden, indim),
        )

    def forward(self, x):
        return self.fc(x)


class ResMLP(nn.Module):
    def __init__(self, indim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(indim, eps=1e-6)
        self.mlp = MLP(indim, hidden)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class Encoder(nn.Module):
    def __init__(self, bs=[64, 128, 256], select_index=None, vq_b=VQ_B, vq_dim=VQ_DIM):
        super().__init__()
        self.n = 462
        if select_index is None:
            select_index = list(range(self.n))
        self.vq_dim = vq_dim
        self.register_buffer("select_index", torch.tensor(select_index).long().unsqueeze(0))
        self.blocks = nn.Sequential(
            nn.Linear(self.n, 512),
            ResMLP(512, 2048),
        )
        self.vq_b = vq_b
        self.bd = {b: i for i, b in enumerate(bs)}
        self.heads = nn.ModuleList([
            nn.Linear(512, b * 8 // self.vq_b * self.vq_dim)
            for b in bs
        ])
        self.vqs = nn.ModuleList([
            VQ(vq_b, vq_dim)
            for _ in bs
        ])

    def forward(self, x, bit):
        b = x.size(0)
        i = self.bd[bit]
        x = x.gather(1, self.select_index.expand(b, -1))
        x = self.blocks(x)
        x = self.heads[i](x).view(b, -1, self.vq_dim)
        x = self.vqs[i].quant(x)
        return x


class Decoder(nn.Module):
    def __init__(self, bs=[64, 128, 256], select_index=None, vq_b=VQ_B, vq_dim=VQ_DIM):
        super().__init__()
        self.n = 462
        if select_index is None:
            select_index = list(range(self.n))
        self.vq_dim = vq_dim
        self.register_buffer("select_index", torch.tensor(select_index).long().unsqueeze(0))
        self.blocks = nn.Sequential(
            ResMLP(512, 2048),
            nn.Linear(512, self.n),
        )
        self.bd = {b: i for i, b in enumerate(bs)}
        self.vq_b = vq_b
        self.heads = nn.ModuleList([
            nn.Linear(b * 8 // self.vq_b * self.vq_dim, 512)
            for b in bs
        ])
        self.vqs = nn.ModuleList([
            VQ(vq_b, vq_dim)
            for _ in bs
        ])

    def forward(self, x, bit):
        b = x.size(0)
        i = self.bd[bit]
        x = self.vqs[i].dequant(x).view(b, -1)
        x = self.heads[i](x)
        x = self.blocks(x)
        y = torch.zeros((b, 2048)).float().to(x.device)
        y.scatter_(1, self.select_index.expand(b, -1), x)
        return y


class TrainModel(nn.Module):
    def __init__(self, bs=[64, 128, 256], select_index=None, vq_b=VQ_B, vq_dim=VQ_DIM):
        super().__init__()
        self.bs = bs
        self.encoder = Encoder(bs, select_index, vq_b, vq_dim)
        self.decoder = Decoder(bs, select_index, vq_b, vq_dim)
        self.decoder.vqs = self.encoder.vqs

    def forward(self, x):
        b = x.size(0)
        x = x.gather(1, self.encoder.select_index.expand(b, -1))
        y = self.encoder.blocks(x)
        losses = dict()
        for bit in self.bs:
            i = self.encoder.bd[bit]
            yi = self.encoder.heads[i](y).view(b, -1, self.encoder.vq_dim)
            qy, loss1, loss2 = self.encoder.vqs[i](yi)
            yi = self.decoder.heads[i](qy.view(b, -1))
            yi = self.decoder.blocks(yi)
            losses[f"{bit}_loss"] = self.cal_loss(x, yi).mean()
            losses[f"{bit}_loss_vq"] = loss1 + loss2 * 0.25
        return losses

    @staticmethod
    def cal_loss(x, y):
        return (x - y).pow(2).sum(-1).sqrt()


if __name__ == "__main__":
    bs = [64, 128, 256]
    select_index = None
    a = Encoder(bs, select_index)
    b = Decoder(bs, select_index)
    c = TrainModel(bs, select_index)
    x = torch.randn((2, 2048))
    for bit in bs:
        x1 = a(x, bit)
        print(x1.shape)
        x2 = b(x1, bit)
        print(x2.shape)

    losses = c(x)
    print(losses)
