import torch
from torch import nn
from torch import optim
from agents.base import BaseAgent
from graphs.models.LiftingBasedDWT_net import LiftingBasedDWTNetWrapper
from graphs.losses.rate_dist import TrainRDLoss, TrainDLoss
from dataloaders.image_dl import ImageDataLoader
from loggers.rate import RateLogger, RDLogger
from utils.image_plots import display_image_in_actual_size, plot_hist_of_rgb_image
from compressai.transforms import RGB2YCbCr, YCbCr2RGB
from graphs.layers.post_processing_networks import DnCNN, PostProcessingiWave, IRCNN, DIDN, DUDnCNN


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
        self.postprocessflag = config.postprocess
        self.optimizer = configure_optimizers(self.model, self.lr)
        if self.config.mode =="train_postprocess":
            if (self.postprocessflag == 'DnCNN'):
                self.postprocess = DnCNN(self.clrch)
            elif (self.postprocessflag == 'iwave'):
                self.postprocess = PostProcessingiWave(config)
            elif (self.postprocessflag == 'IRCNN'):
                self.postprocess = IRCNN(self.clrch, self.clrch)
            elif (self.postprocessflag == 'DIDN'):
                self.postprocess = DIDN(self.config)
            elif (self.postprocessflag == 'DUDnCNN'):
                self.postprocess = DUDnCNN(self.config)
            self.postprocess = self.postprocess.to(self.device)
            self.optimizer_postprocess = optim.Adam(self.postprocess.parameters(), lr = 0.0001)
            self.scheduler_postprocess = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_postprocess, factor=0.5, patience=5,
                                                              threshold=0.0001, threshold_mode='rel',
                                                              cooldown=0, min_lr=1e-06, eps=1e-08, verbose=True)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5,
                                                              threshold=0.0001, threshold_mode='rel',
                                                              cooldown=0, min_lr=1e-06, eps=1e-08, verbose=True)

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
        self.model_size_estimation()

    def train_one_epoch(self):
        self.model.train()
        for batch_idx, x in enumerate(self.data_loader.train_loader):
            self.optimizer.zero_grad()
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
            self.optimizer.step()
            self.current_iteration += 1
            self.train_logger(rd_loss.item(), mse_loss.item(), rate1_loss.item(), rate2_loss.item())
            self.trnit_logger(rd_loss.item(), mse_loss.item(), rate1_loss.item(), rate2_loss.item())

            if (self.current_iteration + 1) % self.loss_prnt_iters == 0:
                trnit_rd_loss, trnit_mse_loss, trnit_rate_loss, _ = self.trnit_logger.display(lr=self.optimizer.param_groups[0]['lr'], typ='it')
                # switch to R+lD loss once training mse for an epoch drops below threshold
                if trnit_mse_loss < self.loss_switch_thr and self.training_loss_switch == 0:
                    self.train_loss = TrainRDLoss(self.lambda_)
                    print("Switching training loss to Rate+lambda*Distortion (it was only lambda*Distortion up to here)")
                    self.training_loss_switch = 1
        train_rd_loss, train_mse_loss, train_rate_loss, _ = self.train_logger.display(lr=self.optimizer.param_groups[0]['lr'], typ='tr')
        self.scheduler.step(train_rd_loss)

    def train_one_epoch_postprocess(self):
        self.model.eval()
        self.postprocess.train()
        for batch_idx, x in enumerate(self.data_loader.train_loader):
            self.optimizer_postprocess.zero_grad()
            x = x.to(self.device)
            if self.clrch == 3:  # rgb img processed together in NN
                # shift pixels to -0.5, 0.5 range
                x = (x - 0.5) * 1
                # run through model, devami asagida
                xhat, self_informations_xe, self_informations_xo_list = self.model(x)
            elif self.clrch == 1:  # yuv image color channels processed separately in NN
                y = self.rgb2ycbcr()(x)
                y = y - torch.tensor([[[0.5]], [[0.0]], [[0.0]]],
                                     device=self.device)  # subract 0.5 from only Y; Cb,Cr already zero-mean
                # run through model, devami asagida
                with torch.no_grad():
                    yhat, self_informations_xe, self_informations_xo_list = self.model(y)
                # yhat = self.postprocess(yhat)
                yhat = yhat + torch.tensor([[[0.5]], [[0.0]], [[0.0]]],
                                           device=self.device)  # add back 0.5 to only Y;
                xhat = self.ycbcr2rgb()(yhat)
                # postprocess the reconstructed rgb image
                xhat = self.postprocess(xhat)
                # shift pixels to -0.5, 0.5 range
                x = (x - 0.5) * 1
                xhat = (xhat - 0.5) * 1
            # calculate loss, back-prop etc.
            rd_loss, mse_loss, rate1_loss, rate2_loss = self.train_loss.forward3(x, xhat, self_informations_xe,
                                                                                 self_informations_xo_list)
            (mse_loss / self.grad_acc_iters).backward()

            self.optimizer_postprocess.step()
            self.current_iteration += 1
            self.train_logger(rd_loss.item(), mse_loss.item(), rate1_loss.item(), rate2_loss.item())
            self.trnit_logger(rd_loss.item(), mse_loss.item(), rate1_loss.item(), rate2_loss.item())

        train_rd_loss, train_mse_loss, train_rate_loss, _ = self.train_logger.display(
            lr=self.optimizer.param_groups[0]['lr'], typ='tr')
        self.scheduler_postprocess.step(train_mse_loss)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            mse_losses = []
            PSNR_val = []
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
                    # display_image_in_actual_size(x, 1, self.device)  # original img
                    display_image_in_actual_size(xhat, 1, self.device)  # reconstructed img
            valid_rd_loss, valid_mse_loss, valid_rate_loss, _ = self.valid_logger.display(lr=0.0, typ='va')

            if self.clrch == 3:
                print(f' avg_psnr = {torch.tensor(PSNR_val).mean().item():.2f}, rate_1 = {torch.tensor(rate_1_val).mean().item()}, rate_2 ={torch.tensor(rate_2_val).mean().item()}, '
                      f'total_rate = {torch.tensor(rate_1_val).mean().item() + torch.tensor(rate_2_val).mean().item()}')
            elif self.clrch == 1:
                print(f' avg_psnr = {torch.tensor(PSNR_val).mean().item():.2f}, rate_1 = {torch.tensor(rate_1_val).mean().item()}, rate_2 ={torch.tensor(rate_2_val).mean().item()}, '
                      f'total_rate = {torch.tensor(rate_1_val).mean().item() + torch.tensor(rate_2_val).mean().item()}')
            return valid_rd_loss

    @torch.no_grad()
    def validate_postprocess(self):
        self.model.eval()
        self.postprocess.eval()
        with torch.no_grad():
            mse_losses = []
            PSNR_val = []
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
                    xhat = self.postprocess(xhat)
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
                    # display_image_in_actual_size(x, 1, self.device)  # original img
                    display_image_in_actual_size(xhat, 1, self.device)  # reconstructed img
            valid_rd_loss, valid_mse_loss, valid_rate_loss, _ = self.valid_logger.display(lr=0.0, typ='va')

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
            mse_losses = []
            PSNR_val = []
            bpp_val = []
            rate_1_val = []
            rate_2_val = []
            for batch_idx, x in enumerate(self.data_loader.test_loader):
                x = x.to(self.device)
                if self.clrch == 3:  # rgb img processed together in NN
                    # shift pixels to -0.5, 0.5 range
                    x = (x - 0.5) * 1
                    # run through model, devami asagida
                    xhat, self_informations_xe, self_informations_xo_list = self.model.compress(x)
                elif self.clrch == 1:  # yuv image color channels processed separately in NN
                    y = self.rgb2ycbcr()(x)
                    y = y - torch.tensor([[[0.5]], [[0.0]], [[0.0]]], device=self.device)  # subract 0.5 from only Y; Cb,Cr already zero-mean
                    # y = y - torch.tensor([[[0.5]], [[cb_mean]], [[cr_mean]]], device=self.device)  # subract 0.5 from only Y
                    # run through model, devami asagida
                    yhat, byte_xe, byte_xo = self.model.compress(y)
                    yhat = yhat + torch.tensor([[[0.5]], [[0.0]], [[0.0]]], device=self.device)  # add back 0.5 to only Y;
                    # yhat = yhat + torch.tensor([[[0.5]], [[cb_mean]], [[cr_mean]]], device=self.device)  # add back 0.5 to only Y;
                    xhat = self.ycbcr2rgb()(yhat)
                    # shift pixels to -0.5, 0.5 range
                    x = (x - 0.5) * 1
                    xhat = (xhat - 0.5) * 1
                # now calculate loss
                # !!! clip xhat to [-0.5 to 0.5], this might give better mse (plotting it gives warnings for clipping)
                xhat.clamp_(-0.5, 0.5)  # this in place version of xhat = torch.clamp(xhat, -0.5, 0.5)

                mse_loss = nn.MSELoss()(x, xhat)
                mse_losses.append(mse_loss.item())
                PSNR_val.append(10.0 * torch.log10(1.0 / torch.tensor(mse_loss.item())))
                rate_1_val.append(byte_xo)
                rate_2_val.append(byte_xe)
                # Plot reconstructed images ?
                if self.imshow_validation:
                    display_image_in_actual_size(x, 1, self.device)  # original img
                    display_image_in_actual_size(xhat, 1, self.device)  # reconstructed img

            if self.clrch == 3:
                print(f' avg_psnr = {torch.tensor(PSNR_val).mean().item():.2f}, rate_high = {torch.tensor(rate_1_val).mean().item()}, rate_low ={torch.tensor(rate_2_val).mean().item()}, '
                      f'total_rate = {torch.tensor(rate_1_val).mean().item() + torch.tensor(rate_2_val).mean().item()}')
            elif self.clrch == 1:
                print(f' avg_psnr = {torch.tensor(PSNR_val).mean().item():.2f}, rate_high = {torch.tensor(rate_1_val).mean().item()}, rate_low ={torch.tensor(rate_2_val).mean().item()}, '
                      f'total_rate = {torch.tensor(rate_1_val).mean().item() + torch.tensor(rate_2_val).mean().item()}')


            return True

    def model_size_estimation(self, print_params=False):
        model = self.model
        if (self.config.postprocess != "none"):
            postprocess = self.postprocess
        if print_params:
            print('---------------Printing paramters--------------------------')
        param_size = 0
        for name, param in model.named_parameters():
            if print_params:
                print(name, type(param), param.size())
            param_size += param.nelement() * param.element_size()
        if (self.config.postprocess !="none"):
            param_size_postprocess = 0
            buffer_size_postprocess = 0
            for name, param in self.postprocess.named_parameters():
                if print_params:
                    print(name, type(param), param.size())
                param_size_postprocess += param.nelement() * param.element_size()
            for name, buffer in self.postprocess.named_buffers():
                if print_params:
                    print(name, type(buffer), buffer.size())
                buffer_size_postprocess += buffer.nelement() * buffer.element_size()
        if print_params:
            print('---------------Printing buffers--------------------------')
        buffer_size = 0
        for name, buffer in model.named_buffers():
            if print_params:
                print(name, type(buffer), buffer.size())
            buffer_size += buffer.nelement() * buffer.element_size()
        if (self.config.postprocess !="none"):
            param_size_mb_postprocess = param_size_postprocess / 1024 ** 2
            buffer_size_mb_postprocess = buffer_size_postprocess / 1024 ** 2
            size_all_mb_postprocess = (param_size_postprocess + buffer_size_postprocess) / 1024 ** 2

        param_size_mb = param_size / 1024 ** 2
        buffer_size_mb = buffer_size / 1024 ** 2
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        if (self.config.postprocess == "none"):
            print('------------------TOT----------------------------------------------')
            print(
                ' model param+buffer=total size: {:.2f}+{:.2f}={:.2f}MB'.format(param_size_mb, buffer_size_mb, size_all_mb))
            print('------------------END----------------------------------------------')
        else:
            print('------------------TOT----------------------------------------------')
            print(
                ' model param+buffer=total size: {:.2f}+{:.2f}={:.2f}MB\n'.format(param_size_mb, buffer_size_mb, size_all_mb))
            print(
                ' postprocess param+buffer=total size: {:.2f}+{:.2f}={:.2f}MB\n'.format(param_size_mb_postprocess, buffer_size_mb_postprocess, size_all_mb_postprocess))
            print(
                ' total param+buffer=total size: {:.2f}+{:.2f}={:.2f}MB\n'.format(param_size_mb_postprocess+param_size_mb,
                                                                                  buffer_size_mb_postprocess+ buffer_size_mb,
                                                                                  size_all_mb_postprocess+ size_all_mb))

            print('------------------END----------------------------------------------')


def configure_optimizers(net, lr):


    # parameters = net.parameters
    parameters = {
        n
        for n, p in net.named_parameters()
        if p.requires_grad
    }
    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())



    parameters_list = list((params_dict[n] for n in sorted(parameters)))

    optimizer = optim.Adam(
        [{'params': parameters_list, 'lr': lr}]
    )

    return optimizer







def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
