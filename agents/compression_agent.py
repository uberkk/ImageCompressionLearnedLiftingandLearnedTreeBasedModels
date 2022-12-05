import torch
from torch import nn
from torch import optim
from visdom import Visdom

from agents.base import BaseAgent
from dataloaders.image_dl import ImageDataLoader
from graphs.losses.rate_dist import TrainRDLoss, ValidRDLoss
from loggers.rate_dist import RDTrainLogger, RDValidLogger


class CompressionAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model        = None
        self.postprocess = None
        self.data_loader  = ImageDataLoader(config)
        self.train_loss   = TrainRDLoss(config.lambda_)
        self.valid_loss   = ValidRDLoss(config.lambda_)
        self.train_logger = RDTrainLogger()
        self.valid_logger = RDValidLogger(config)
        self.lr = self.config.learning_rate
        self.optimizer = None
        self.viz = Visdom(raise_exceptions=True)

    def train_one_epoch(self):
        self.model.train()
        for batch_idx, x in enumerate(self.data_loader.train_loader):
            self.model.entropy.find_cdf_range()
            x = x.to(self.device)
            self.optimizer.zero_grad()
            x_hat, rate = self.model(x)
            loss, mse, rate = self.train_loss(x, x_hat, rate)
            loss.backward()
            self.optimizer.step()
            self.current_iteration += 1
            self.train_logger(loss.item(), mse.item(), rate.item())
        self.train_logger.display()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        if self.model.entropy.find_cdf_range():
            with torch.no_grad():
                self.model.entropy.quantize_cdf()
                for batch_idx, x in enumerate(self.data_loader.valid_loader):
                    x = x.to(self.device)
                    x_hat, rate = self.model(x)
                    loss, psnr, bits = self.valid_loss(x, x_hat, rate)
                    self.valid_logger(loss.item(), psnr.item(), bits.item())
                self.model.display(x)
                self.valid_logger.display()
                self.model.entropy.display()
                return 1/loss.item()
        else:
            return np.inf



        