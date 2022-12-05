import logging
import sys
import shutil

import torch
from torch.backends import cudnn
from utils.mailer import Mailer

cudnn.benchmark = True
cudnn.enabled = True


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.best_valid_loss = float('inf')  # 0
        self.current_epoch = 0
        self.current_iteration = 0
        if config.mode != "test":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.cuda = torch.cuda.is_available() & self.config.cuda
        self.manual_seed = self.config.seed
        self.lr = config.learning_rate
        torch.cuda.manual_seed(self.manual_seed)
        torch.cuda.set_device(self.config.gpu_device)
        
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError
    def train_one_epoch_postprocess(self):
        """
        One epoch of training
        :return:
        """

    def validate(self):
        """
        One cycle of model validation without recursive reconstrcution, where zhat is simply a noisy version of original
        :return:
        """
        raise NotImplementedError

    def validate_recu_reco(self):
        """
        One cycle of model validation with recursive reconstruction, i.e zhat is actual reconstruction
        :return:
        """
        raise NotImplementedError

    def test(self):
        """
        One cycle of model test
        :return:
        """
        raise NotImplementedError

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            # checkpoint = torch.load(filename)
            checkpoint = torch.load(filename, map_location=self.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.best_valid_loss = checkpoint['best_valid_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.train_logger.load_state_dict(checkpoint['train_logger'])
            self.trnit_logger.load_state_dict(checkpoint['trnit_logger'])
            self.valid_logger.load_state_dict(checkpoint['valid_logger'])
            self.test_logger.load_state_dict(checkpoint['test_logger'])
            self.logger.info("Checkpoint loaded successfully \
                            from '{}' at (epoch {}) at (iteration {})\n"
                            .format(self.config.checkpoint_dir, checkpoint['epoch'],
                            checkpoint['iteration']))

            self.model.to(self.device)
            # Fix the optimizer cuda error
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping..."
                             .format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")
            
    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        if(self.config.mode == "train"):
            state = {
                'epoch': self.current_epoch,
                'iteration': self.current_iteration,
                'best_valid_loss': self.best_valid_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'train_logger' : self.train_logger.state_dict(),
                'trnit_logger' : self.trnit_logger.state_dict(),
                'valid_logger' : self.valid_logger.state_dict(),
                'test_logger' : self.test_logger.state_dict()
            }
        elif(self.config.mode == "train_postprocess"):
            state = {
                'epoch': self.current_epoch,
                'iteration': self.current_iteration,
                'best_valid_loss': self.best_valid_loss,
                'state_dict': self.model.state_dict(),
                'state_dict_postprocess': self.postprocess.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'train_logger' : self.train_logger.state_dict(),
                'trnit_logger' : self.trnit_logger.state_dict(),
                'valid_logger' : self.valid_logger.state_dict(),
                'test_logger' : self.test_logger.state_dict()
            }
        torch.save(state, self.config.checkpoint_dir + filename)
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        try:
            if self.config.mode == 'test':
                self.test()
            elif self.config.mode == 'validate':
                self.validate()
            elif self.config.mode == 'validate_recu_reco':
                self.validate_recu_reco()
            elif self.config.mode == 'train':
                self.train()
            elif self.config.mode == 'train_postprocess':
                self.train_postprocess()
            elif self.config.mode == 'debug':
                with torch.autograd.detect_anomaly():
                    self.train()
            else:
                raise NameError("'" + self.config.mode + "'" 
                                + ' is not a valid training mode.' )
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
        except AssertionError as e:
            raise e
        except Exception as e:
            self.save_checkpoint()
            raise e

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            if not (self.current_epoch+1) % self.config.validate_every:
                valid_loss = self.validate()
                is_best = valid_loss < self.best_valid_loss
                if is_best:
                    self.best_valid_loss = valid_loss
                self.save_checkpoint(is_best=is_best)
            # if not (self.current_epoch+1) % self.config.test_every:
            #     test_loss = self.test()
            self.current_epoch += 1

    def train_postprocess(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch_postprocess()
            if not (self.current_epoch+1) % self.config.validate_every:
                valid_loss = self.validate_postprocess()
                is_best = valid_loss < self.best_valid_loss
                if is_best:
                    self.best_valid_loss = valid_loss
                self.save_checkpoint(is_best=is_best)
            # if not (self.current_epoch+1) % self.config.test_every:
            #     test_loss = self.test()
            self.current_epoch += 1

    def finalize(self):
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        if (self.config.mode =="train" or self.config.mode == "train_postprocess"):
            self.save_checkpoint()
