import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from visdom import Visdom

from agents.base import BaseAgent
from graphs.models.LiftingBasedDWT_net import LiftingBasedDWTNetWrapper
from graphs.losses.rate_dist import TrainRDLoss, TrainDLoss
# from graphs.losses.rate_distortion_loss import TrainRateLoss, ValidRateLoss
from dataloaders.image_dl import ImageDataLoader
# from loggers.rate_dist import RDTrainLogger, RDValidLogger
from loggers.rate import RateLogger, RDLogger
from utils.image_plots import display_image_in_actual_size, plot_hist_of_rgb_image
from compressai.transforms import RGB2YCbCr, YCbCr2RGB



class LiftingBasedDWTAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # get comressai's color transformations
        self.clrch = config.clrch
        self.rgb2ycbcr = RGB2YCbCr
        self.ycbcr2rgb = YCbCr2RGB
        self.lr = self.config.learning_rate
        self.model = LiftingBasedDWTNetWrapper(config)
        self.model = self.model.to(self.device)
        self.optimizer, self.aux_optimizer = configure_optimizers(self.model, self.lr, self.lr * 10)
        # self.optimizer = optim.Adam([{'params': self.model.parameters(), 'lr':self.lr}])
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=config.gamma, verbose=False)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[400, 500], gamma=0.1, verbose=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=30,
                                                              threshold=0.0001, threshold_mode='rel',
                                                              cooldown=0, min_lr=1e-05, eps=1e-08, verbose=True)
        self.grad_acc_iters = config.grad_acc_iters
        self.loss_prnt_iters = config.loss_prnt_iters
        self.data_loader  = ImageDataLoader(config)
        self.lambda_      = config.lambda_
        self.loss_switch_thr = config.loss_switch_thr
        self.training_loss_switch = config.training_loss_switch
        if self.training_loss_switch == 0:
            self.train_loss = TrainDLoss(config.lambda_)
            print("Starting training with only lambda*Distortion training loss...")
        else:
            self.train_loss   = TrainRDLoss(config.lambda_)
        self.valid_loss   = TrainRDLoss(config.lambda_)
        self.train_logger = RDLogger()
        self.trnit_logger = RDLogger() # to report for every 1000 iterations inside and epoch
        self.aux_logger   = RDLogger()
        self.valid_logger = RDLogger()
        self.test_logger  = RDLogger()
        # self.viz = Visdom(raise_exceptions=True)
        if config.mode == 'test' or config.mode == 'validate' or config.mode == 'validate_recu_reco':
            self.load_checkpoint('model_best.pth.tar')
            self.imshow_validation = config.imshow_validation
        elif config.resume_training:
            self.load_checkpoint(self.config.checkpoint_file)
            self.imshow_validation = False
        else:
            self.imshow_validation = False

    def train_one_epoch(self):
        self.model.train()
        for batch_idx, x in enumerate(self.data_loader.train_loader):
            x = x.to(self.device)
            if self.clrch == 3:  # rgb img processed together in NN
                # shift pixels to -0.5, 0.5 range
                x = (x - 0.5)*1
                # run through model, devami asagida
                xhat, self_informations_xe, self_informations_xo_list = self.model(x)
            elif self.clrch == 1:  # yuv image color channels processed separately in NN
                y = self.rgb2ycbcr()(x)
                y = y - torch.tensor([[[0.5]], [[0.0]], [[0.0]]], device=self.device)  # subract 0.5 from only Y; Cb,Cr already zero-mean
                # run through model, devami asagida
                yhat, self_informations_xe, self_informations_xo_list = self.model(y)
                yhat = yhat + torch.tensor([[[0.5]], [[0.0]], [[0.0]]], device=self.device)  # add back 0.5 to only Y;
                xhat = self.ycbcr2rgb()(yhat)
                # shift pixels to -0.5, 0.5 range
                x = (x - 0.5) * 1
                xhat = (xhat - 0.5) * 1
            # calculate loss, back-prop etc.
            rd_loss, mse_loss, rate1_loss, rate2_loss = self.train_loss.forward3(x, xhat, self_informations_xe, self_informations_xo_list)
            (rd_loss / self.grad_acc_iters).backward()
            # aux loss
            if not (self.aux_optimizer is None):
                aux_loss = self.model.aux_loss()
                (aux_loss / self.grad_acc_iters).backward()
            # gradeint accumulation of grad_acc_iters
            if ((self.current_iteration + 1) % self.grad_acc_iters == 0) or ((batch_idx+1) == len(self.data_loader.train_loader)):
                # apply gradient clipping/scaling (if loss has been switched to R+lD)
                #if True: # False and self.training_loss_switch == 1:
                    # https://github.com/liujiaheng/compression/blob/master/train.py
                    # clip_gradient(self.optimizer, 1.0) # 0.1  5.0
                    # https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
                    # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0) # 0.1
                    #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)
                # update weights, iteration number and log losses
                self.optimizer.step()
                self.optimizer.zero_grad()
                if not (self.aux_optimizer is None):
                    self.aux_optimizer.step()
                    self.aux_optimizer.zero_grad()
            self.current_iteration += 1
            self.train_logger(rd_loss.item(), mse_loss.item(), rate1_loss.item(), rate2_loss.item())
            self.trnit_logger(rd_loss.item(), mse_loss.item(), rate1_loss.item(), rate2_loss.item())
            if not (self.aux_optimizer is None):
                self.aux_logger(aux_loss.item(), 0.0, 0.0, 0.0)
            if (self.current_iteration + 1) % self.loss_prnt_iters == 0:
                trnit_rd_loss, trnit_mse_loss, trnit_rate_loss, _ = self.trnit_logger.display(lr=self.optimizer.param_groups[0]['lr'], typ='it')
                # switch to R+lD loss once training mse for an epoch drops below threshold
                if trnit_mse_loss < self.loss_switch_thr and self.training_loss_switch == 0:
                    self.train_loss = TrainRDLoss(self.lambda_)
                    print("Switching training loss to Rate+lambda*Distortion (it was only lambda*Distortion up to here)")
                    self.training_loss_switch = 1
        train_rd_loss, train_mse_loss, train_rate_loss, _ = self.train_logger.display(lr=self.optimizer.param_groups[0]['lr'], typ='tr')
        if not (self.aux_optimizer is None):
            aux_rd_loss, _, _, _ = self.aux_logger.display(lr=self.optimizer.param_groups[0]['lr'], typ='tr')

        self.scheduler.step(train_rd_loss)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            mse_losses = []
            PSNR_val = []
            bpp_val = []
            rate_1_val = []
            rate_2_val = []
            for batch_idx, x in enumerate(self.data_loader.valid_loader):
                x = x.to(self.device)
                if self.clrch == 3:  # rgb img processed together in NN
                    # shift pixels to -0.5, 0.5 range
                    x = (x - 0.5) * 1
                    # run through model, devami asagida
                    xhat, self_informations_xe, self_informations_xo_list = self.model(x)
                elif self.clrch == 1:  # yuv image color channels processed separately in NN
                    y = self.rgb2ycbcr()(x)
                    y = y - torch.tensor([[[0.5]], [[0.0]], [[0.0]]], device=self.device)  # subract 0.5 from only Y; Cb,Cr already zero-mean
                    # run through model, devami asagida
                    yhat, self_informations_xe, self_informations_xo_list = self.model(y)
                    yhat = yhat + torch.tensor([[[0.5]], [[0.0]], [[0.0]]], device=self.device)  # add back 0.5 to only Y;
                    xhat = self.ycbcr2rgb()(yhat)
                    # shift pixels to -0.5, 0.5 range
                    x = (x - 0.5) * 1
                    xhat = (xhat - 0.5) * 1
                # now calculate loss
                # !!! clip xhat to [-0.5 to 0.5], this might give better mse (plotting it gives warnings for clipping)
                xhat.clamp_(-0.5, 0.5)  # this in place version of xhat = torch.clamp(xhat, -0.5, 0.5)
                rd_loss, mse_loss, rate1_loss, rate2_loss = self.valid_loss.forward3(x, xhat, self_informations_xe,
                                                                                     self_informations_xo_list)
                self.valid_logger(rd_loss.item(), mse_loss.item(), rate1_loss.item(), rate2_loss.item())
                mse_losses.append(mse_loss.item())
                PSNR_val.append(10.0 * torch.log10(1.0 / torch.tensor(mse_loss.item())))
                rate_1_val.append(rate1_loss)
                rate_2_val.append(rate2_loss)
                # Plot reconstructed images ?
                if self.imshow_validation:
                    display_image_in_actual_size(x, 1, self.device)  # original img
                    display_image_in_actual_size(xhat, 1, self.device)  # reconstructed img
            valid_rd_loss, valid_mse_loss, valid_rate_loss, _ = self.valid_logger.display(lr=0.0, typ='va')

            self.scheduler.step(valid_rd_loss)

            # if self.clrch == 3:
            #     print(f'  avg_psnr = {10.0 * torch.log10(1.0 / torch.tensor(mse_losses)).mean().item():.2f}')
            # elif self.clrch == 1:
            #     print(f' avg_psnr = {10.0 * torch.log10(1.0 / torch.tensor(mse_losses)).mean().item():.2f}')

            if self.clrch == 3:
                print(f' avg_psnr = {torch.tensor(PSNR_val).mean().item():.2f}, rate_1 = {torch.tensor(rate_1_val).mean().item()}, rate_2 ={torch.tensor(rate_2_val).mean().item()}, '
                      f'total_rate = {torch.tensor(rate_1_val).mean().item() + torch.tensor(rate_2_val).mean().item()}')
            elif self.clrch == 1:
                print(f' avg_psnr = {torch.tensor(PSNR_val).mean().item():.2f}, rate_1 = {torch.tensor(rate_1_val).mean().item()}, rate_2 ={torch.tensor(rate_2_val).mean().item()}, '
                      f'total_rate = {torch.tensor(rate_1_val).mean().item() + torch.tensor(rate_2_val).mean().item()}')


            return valid_rd_loss

    @torch.no_grad()
    def validate_recu_reco(self):
        self.model.eval()
        with torch.no_grad():
            pass

    # test should be modified to have actual entorpy coding....
    @torch.no_grad()
    def test(self):
        self.model.eval()
        with torch.no_grad():
            pass


from itertools import chain


def configure_optimizers(net, lr, aux_lr):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and not (("scl_" in n) or ("scb_" in n)) and p.requires_grad
    }
    scl_parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and (("scl_" in n) or ("scb_" in n)) and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters & scl_parameters
    union_params = parameters | aux_parameters | scl_parameters

    #assert len(inter_params) == 0
    #assert len(union_params) - len(params_dict.keys()) == 0

    parameters_list = list((params_dict[n] for n in sorted(parameters)))
    scl_parameters_list = list((params_dict[n] for n in sorted(scl_parameters)))
    optimizer = optim.Adam(
        [{'params': parameters_list, 'lr': lr},
         {'params': scl_parameters_list, 'lr': lr*10}]
    )
    if len(aux_parameters) > 0:
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=aux_lr,
        )
        return optimizer, aux_optimizer
    else:
        return optimizer, None


def configure_optimizers3(net, lr, aux_lr, net1, net2):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    par_list, scl_par_list, aux_par_list = get_par_list_for_opt(net)
    par1_list, scl_par1_list, aux_par1_list = get_par_list_for_opt(net1)
    par2_list, scl_par2_list, aux_par2_list = get_par_list_for_opt(net2)

    optimizer = optim.Adam(
        [{'params': par_list, 'lr': lr},
         {'params': par1_list, 'lr': lr},
         {'params': par2_list, 'lr': lr},
         {'params': scl_par_list, 'lr': lr*10},
         {'params': scl_par1_list, 'lr': lr*10},
         {'params': scl_par2_list, 'lr': lr*10},
        ]
    )
    aux_optimizer = optim.Adam(
        [{'params': aux_par_list, 'lr': aux_lr},
         {'params': aux_par1_list, 'lr': aux_lr},
         {'params': aux_par2_list, 'lr': aux_lr},
        ]
    )
    return optimizer, aux_optimizer


def get_par_list_for_opt(net):
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and not (("scl_" in n) or ("scb_" in n)) and p.requires_grad
    }
    scl_parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and (("scl_" in n) or ("scb_" in n)) and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters & scl_parameters
    union_params = parameters | aux_parameters | scl_parameters

    #assert len(inter_params) == 0
    #assert len(union_params) - len(params_dict.keys()) == 0

    parameters_list = list((params_dict[n] for n in sorted(parameters)))
    scl_parameters_list = list((params_dict[n] for n in sorted(scl_parameters)))
    aux_parameters_list = list((params_dict[n] for n in sorted(aux_parameters)))

    return parameters_list, scl_parameters_list, aux_parameters_list


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
