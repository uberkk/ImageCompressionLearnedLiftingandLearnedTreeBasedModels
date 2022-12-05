import logging
import json
import os
from statistics import mean

import torch
import numpy as np

from visdom import Visdom
from torchnet.logger import visdomlogger as vlog


class RateDistortionMeter():
    def __init__(self):
        self.loss    = []
        self.mse     = []
        self.rate    = []
        self.current_iteration = 0
        self.current_epoch     = 0

    def append(self, loss, mse, rate):
        self.current_iteration += 1
        self.loss.append(loss)
        self.mse.append(mse)
        self.rate.append(rate)

    def reset(self):
        self.loss    = []
        self.mse     = []
        self.rate    = []

    def mean(self):
        self.current_epoch += 1
        loss = mean(self.loss)
        mse  = mean(self.mse)
        rate = mean(self.rate)
        self.reset()
        return loss, mse, rate


class RDTrainLogger(RateDistortionMeter):
    def __init__(self):
        super(RDTrainLogger, self).__init__()
        # self.viz = Visdom(raise_exceptions=True)
        self.logger = logging.getLogger("Loss")
        # self.loss_logger = vlog.VisdomPlotLogger('line', opts={'title': 'Loss'})

    def __call__(self, *args):
        self.append(*args)

    def display(self):
        loss, mse, rate = self.mean()
        self.text_log(self.current_epoch, loss, mse, rate)
        # self.visdom_log(self.current_epoch, loss)

    # def visdom_log(self, cur_iter, loss):
    #     self.loss_logger.log(cur_iter, loss, name="train")

    def text_log(self, cur_iter, loss, mse, rate):
        self.logger.info(
            'Train Epoch: {} Avg. Loss: {:.4f} MSE: {:.4f}  Rate: {:.2f}'
            .format(cur_iter, loss, mse, rate))


class RDValidLogger(RateDistortionMeter):
    def __init__(self, config):
        super(RDValidLogger, self).__init__()
        self.lambda_ = config.lambda_
        self.result_file = os.path.join('experiments', 
                                        config.multi_exp_name, 
                                        'results.json')

        self.viz = Visdom(raise_exceptions=True)
        self.logger = logging.getLogger("Loss")
        self.bpp_logger = vlog.VisdomPlotLogger('line', opts={'title': 'bpp'})
        self.psnr_logger = vlog.VisdomPlotLogger('line', opts={'title': 'PSNR'})

    def __call__(self, *args):
        self.append(*args)

    def display(self):
        loss, psnr, bpp = self.mean()
        self.text_log(psnr, bpp)
        self.visdom_log(self.current_epoch, psnr, bpp)
        self.json_log(psnr, bpp)

    def visdom_log(self, cur_iter, psnr, bpp):
        self.bpp_logger.log(cur_iter, bpp)
        self.psnr_logger.log(cur_iter, psnr)

    def text_log(self, psnr, bpp):
        self.logger.info(
            'Valid Avg. PSNR: {:.1f}  bpp: {:.2f}'
            .format(psnr, bpp))

    def json_log(self, mse, bpp):
        try:
            f = open(self.result_file, 'r+')
        except FileNotFoundError:
            f = open(self.result_file, 'w')
            f.seek(0)
            d = {}
            json.dump(d, f, indent=4)
            f.truncate()
            f.close()
            f = open(self.result_file, 'r+')

        result = {'dist':mse, 'rate':bpp}
        data = json.load(f)
        data[str(self.lambda_)] = result
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
        f.close()