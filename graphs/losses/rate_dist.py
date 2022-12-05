import logging
import json
import os
from statistics import mean

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from visdom import Visdom


class TrainRDLoss(nn.Module):
    def __init__(self, lambda_):
        super(TrainRDLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.lambda_ = lambda_

    def forward(self, x, x_hat, rate):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate = torch.sum(rate) / torch.numel(x) * 3
        # self.loss = self.mse + self.lambda_ * self.rate
        self.loss = self.rate + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate

    def forward2(self, x, x_hat, rate1, rate2):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = torch.sum(rate2) / torch.numel(x) * 3
        # self.loss = self.mse + self.lambda_ * self.rate
        self.loss = self.rate1 + self.rate2 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2

    def forward3(self, x, x_hat, rate1, rate2list):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = 0
        for i in range(len(rate2list)):
            self.rate2 += torch.sum(rate2list[i]) / torch.numel(x) * 3
        self.loss = self.rate1 + self.rate2 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2


class TrainDLoss(TrainRDLoss):
    def __init__(self, lambda_):
        super(TrainDLoss, self).__init__(lambda_)

    def forward(self, x, x_hat, rate):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate = torch.sum(rate) / torch.numel(x) * 3
        # self.loss = self.mse + self.lambda_ * self.rate
        self.loss = 0         + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate

    def forward2(self, x, x_hat, rate1, rate2):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = torch.sum(rate2) / torch.numel(x) * 3
        # self.loss = self.rate1 + self.rate2 + self.lambda_ * self.mse
        self.loss = 0 + 0 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2

    def forward3(self, x, x_hat, rate1, rate2list):
        self.mse  = self.mse_loss(x, x_hat)
        self.rate1 = torch.sum(rate1) / torch.numel(x) * 3
        self.rate2 = 0
        for i in range(len(rate2list)):
            self.rate2 += torch.sum(rate2list[i]) / torch.numel(x) * 3
        self.loss = 0 + 0 + self.lambda_ * self.mse
        return self.loss, self.mse, self.rate1, self.rate2


class ValidRDLoss(nn.Module):
    def __init__(self, lambda_):
        super(ValidRDLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x, x_hat, rate):
        self.mse  = self.psnr(x, x_hat)
        if type(rate) == int:
            rate = torch.tensor([float(rate)])
            # rate = float(rate)
            # self.rate = rate  / torch.numel(x) * 3
        self.rate = torch.sum(rate, dtype=torch.float) / torch.numel(x) * 3
        self.loss = self.mse + self.rate*self.lambda_
        return self.loss, self.mse, self.rate

    def psnr(self, x, x_hat):
        mse  = F.mse_loss(x_hat, x, reduction='none')
        mse  = torch.mean(mse.view(mse.shape[0], -1), 1)
        psnr = -10*torch.log10(mse)
        psnr = torch.mean(psnr)
        return psnr