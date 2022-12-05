import logging
from statistics import mean

import torch
from datetime import datetime


class RateMeter():
    def __init__(self):
        self.rate    = []
        self.current_iteration = 0
        self.current_epoch     = 0

    def append(self, rate):
        self.current_iteration += 1
        self.rate.append(rate)

    def reset(self):
        self.rate    = []

    def mean(self):
        self.current_epoch += 1
        rate = mean(self.rate)
        self.reset()
        return rate


class RateLogger(RateMeter):
    def __init__(self):
        super(RateLogger, self).__init__()
        self.logger = logging.getLogger("Loss")

    def __call__(self, *args):
        self.append(*args)

    def display(self, lr=0.0):
        rate = self.mean()
        self.text_log(self.current_epoch, rate, lr)
        return rate

    def text_log(self, cur_iter, rate, lr):
        if lr > 1e-13:
            self.logger.info('Train Epoch: {:3d} Rate: {:.2f}  (lr: {:.8f})'.format(cur_iter, rate, lr))
        elif lr < -1e-13:
            self.logger.info('           Test Epoch: {:3d} Rate: {:.2f} '.format(cur_iter, rate))
        else:
            self.logger.info('          Valid Epoch: {:3d} Rate: {:.2f} '.format(cur_iter, rate))


class RateDistortionMeter():
    def __init__(self):
        self.loss    = []
        self.mse     = []
        self.rate    = []
        self.rate2   = []
        self.current_iteration = 0
        self.current_epoch     = 0

    def append(self, loss, mse, rate, rate2=0):
        self.current_iteration += 1
        self.loss.append(loss)
        self.mse.append(mse)
        self.rate.append(rate)
        if rate2 > 0:
            self.rate2.append(rate2)

    def reset(self):
        self.loss    = []
        self.mse     = []
        self.rate    = []
        self.rate2   = []

    def mean(self):
        self.current_epoch += 1
        loss = mean(self.loss)
        mse  = mean(self.mse)
        rate = mean(self.rate)
        if len(self.rate2) > 0:
            rate2 = mean(self.rate2)
        else:
            rate2 = 0
        self.reset()
        return loss, mse, rate, rate2

    def state_dict(self):
        return {'loss':self.loss, 'mse':self.mse, 'rate':self.rate, 'rate2':self.rate2, 'it':self.current_iteration, 'ep':self.current_epoch}

    def load_state_dict(self, info):
        self.loss    = info['loss']
        self.mse     = info['mse']
        self.rate    = info['rate']
        self.rate2   = info['rate2']
        self.current_iteration = info['it']
        self.current_epoch     = info['ep']


class RDLogger(RateDistortionMeter):
    def __init__(self):
        super(RDLogger, self).__init__()
        # self.viz = Visdom(raise_exceptions=True)
        self.logger = logging.getLogger("Loss")
        # self.loss_logger = vlog.VisdomPlotLogger('line', opts={'title': 'Loss'})
        self.t1 = datetime.now()
        self.t2 = datetime.now()

    def __call__(self, *args):
        self.append(*args)

    def display(self, lr=0.0, typ='tr'):
        loss, mse, rate, rate2 = self.mean()
        self.text_log(self.current_epoch, loss, mse, rate, rate2, lr, typ)
        # self.visdom_log(self.current_epoch, loss)
        return loss, mse, rate, rate2

    # def visdom_log(self, cur_iter, loss):
    #     self.loss_logger.log(cur_iter, loss, name="train")

    def text_log(self, cur_iter, loss, mse, rate, rate2, lr, typ):
        psnr = 10*torch.log10(torch.Tensor([1.0**2])/mse).item()
        # timedifstr = self._get_time_diff_str()
        timedifstr = self._get_time_now_str()
        if rate2 < 10**-6:
            # self.logger.info('Train Epoch: {} Avg. Loss: {:.4f} MSE: {:.4f}  Rate: {:.2f}'.format(cur_iter, loss, mse, rate))
            if typ == 'tr':
                self.logger.info('  Train Epoch: {:3d}  RDLoss: {:.6f} MSE/PSNR: {:.6f}/{:.2f} Rate: {:.3f}  (lr: {:.6f}) ({})'.format(cur_iter, loss, mse, psnr, rate, lr, timedifstr))
            elif typ == 'te':
                self.logger.info('   Test Epoch: {:3d}  RDLoss: {:.6f} MSE/PSNR: {:.6f}/{:.2f} Rate: {:.3f} ({})'.format(cur_iter, loss, mse, psnr, rate, timedifstr))
            elif typ == 'va':
                self.logger.info('  Valid Epoch: {:3d}  RDLoss: {:.6f} MSE/PSNR: {:.6f}/{:.2f} Rate: {:.3f} ({})'.format(cur_iter, loss, mse, psnr, rate, timedifstr))
            elif typ == 'it':
                self.logger.info('Train Itera: {:3d}  RDLoss: {:.6f} MSE/PSNR: {:.6f}/{:.2f} Rate: {:.3f}  (lr: {:.6f}) ({})'.format(cur_iter, loss, mse, psnr, rate, lr, timedifstr))
        else:
            if typ == 'tr':
                self.logger.info('  Train Epoch: {:3d}  RDLoss: {:.6f} MSE/PSNR: {:.6f}/{:.2f} Rate: {:.3f}+{:.3f}  (lr: {:.6f}) ({})'.format(cur_iter, loss, mse, psnr, rate, rate2, lr, timedifstr))
            elif typ == 'te':
                self.logger.info('   Test Epoch: {:3d}  RDLoss: {:.6f} MSE/PSNR: {:.6f}/{:.2f} Rate: {:.3f}+{:.3f} ({})'.format(cur_iter, loss, mse, psnr, rate, rate2, timedifstr))
            elif typ == 'va':
                self.logger.info('  Valid Epoch: {:3d}  RDLoss: {:.6f} MSE/PSNR: {:.6f}/{:.2f} Rate: {:.3f}+{:.3f} ({})'.format(cur_iter, loss, mse, psnr, rate, rate2, timedifstr))
            elif typ == 'it':
                self.logger.info('Train Itera: {:3d}  RDLoss: {:.6f} MSE/PSNR: {:.6f}/{:.2f} Rate: {:.3f}+{:.3f}  (lr: {:.6f}) ({})'.format(cur_iter, loss, mse, psnr, rate, rate2, lr, timedifstr))

    def _get_time_diff_str(self):
        self.t2 = datetime.now()
        diff = self.t2 - self.t1
        self.t1 = self.t2 # update t1 for next time
        diffstr = str(diff).split(".")[0]
        return diffstr

    def _get_time_now_str(self):
        self.t2 = datetime.now()
        return self.t2.strftime("%H:%M:%S")
