import torch.nn as nn
import torch
lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781,
                 1.149604398860241]  # bior4.4
class wavelet_inverse_v2(nn.Module):
    def __init__(self, P, U,resnet_coeff, liftingLevel, convBlockList, cfg, nh=0, nl=0):
        super(wavelet_inverse_v2, self).__init__()
        self.P = P
        self.U = U
        self.lifting_level = liftingLevel
        self.resnet_coeff = resnet_coeff
        self.convBlock = convBlockList
        self.csize = cfg.clrch
        self.scale = cfg.scale
        self.nh = nh
        self.nl = nl
        self.config = cfg


    def one_level_lifting(self, LL, LH, HL, HH):
        LL = torch.transpose(LL, 2, 3)
        HL = torch.transpose(HL, 2, 3)

        LL, HL = self.lifting_inverse_row_2_stage_lifting(LL, HL)

        L = self.reconstruct_fun(LL, HL)

        LH = torch.transpose(LH, 2, 3)
        HH = torch.transpose(HH, 2, 3)

        LH, HH = self.lifting_inverse_row_2_stage_lifting(LH, HH)

        H = self.reconstruct_fun(LH, HH)

        L, H = self.lifting_inverse_row_2_stage_lifting(L, H)
        recon = self.reconstruct_fun(L, H)
        recon = torch.transpose(recon, 2, 3)
        return recon

    def reconstruct_fun(self, up, bot):
        # temp_L = torch.transpose(up, 2, 3)
        # temp_H = torch.transpose(bot, 2, 3)
        x_shape = up.size()
        x_n = x_shape[0]
        x_c = x_shape[1]
        x_w = x_shape[2]
        x_h = x_shape[3]
        if self.config.mode != "test":
            recon = torch.FloatTensor(x_n, x_c, x_w * 2, x_h).to('cuda')
        else:
            recon = torch.FloatTensor(x_n, x_c, x_w * 2, x_h)
        recon[:, :, 0::2, :] = up
        recon[:, :, 1::2, :] = bot

        recon = torch.transpose(recon, 2, 3)
        return recon


    def lifting_inverse_row(self, L, H):
        for index in range(self.lifting_level-1,-1,-1):
            skip, H_net = self.U[index](H)
            L = L - skip - H_net * self.resnet_coeff

            skip, L_net = self.P[index](L)
            H = H - skip - L_net * self.resnet_coeff
        return L, H

    def lifting_inverse_row_2_stage_lifting(self, L, H):

        if(self.scale==1):
            nh = lifting_coeff[4] + self.nh * 0.1
            nl = lifting_coeff[5] + self.nl * 0.1
            H = H / nh
            L = L / nl

        skip = self.convBlock[3](H)
        H_net = self.U[1](skip)
        L = L - skip - H_net * self.resnet_coeff

        skip = self.convBlock[2](L)
        L_net = self.P[1](skip)
        H = H - skip - L_net * self.resnet_coeff

        skip = self.convBlock[1](H)
        H_net = self.U[0](skip)
        L = L - skip - H_net * self.resnet_coeff

        skip = self.convBlock[0](L)
        L_net = self.P[0](skip)
        H = H - skip - L_net * self.resnet_coeff

        return L,H