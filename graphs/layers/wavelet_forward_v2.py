# import torch.nn as nn
# import torch
#
lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781,
                 1.149604398860241]  # bior4.4


import torch.nn as nn
import torch

class wavelet_forward_v2(nn.Module):
    def __init__(self, P, U ,resnet_coeff, liftingLevel,convBlockList, cfg, nh=0, nl=0):
        super(wavelet_forward_v2, self).__init__()
        self.P = P
        self.U = U
        self.resnet_weight = resnet_coeff
        self.lifting_level = liftingLevel
        self.csize = cfg.clrch
        self.convBlock = convBlockList
        self.nh = nh
        self.nl = nl
        self.scale = cfg.scale



    def one_level_lifting(self, x):
        L = x[:, :, 0::2, :]
        H = x[:, :, 1::2, :]
        L, H = self.lifting_forward_row_2_stage_lifting(L, H)

        # step 2: for w, L
        L = torch.transpose(L, 2, 3)
        LL = L[:, :, 0::2, :]
        HL = L[:, :, 1::2, :]

        LL, HL = self.lifting_forward_row_2_stage_lifting(LL, HL)

        LL = torch.transpose(LL, 2, 3)
        HL = torch.transpose(HL, 2, 3)

        # step 2: for w, H

        H = torch.transpose(H, 2, 3)

        LH = H[:, :, 0::2, :]
        HH = H[:, :, 1::2, :]

        LH, HH = self.lifting_forward_row_2_stage_lifting(LH, HH)

        LH = torch.transpose(LH, 2, 3)
        HH = torch.transpose(HH, 2, 3)


        return LL, LH, HL, HH



    def lifting_forward_row_2_stage_lifting(self, L, H):

        skip = self.convBlock[0](L)
        L_net = self.P[0](skip)
        H = H + skip + L_net * self.resnet_weight

        skip = self.convBlock[1](H)
        H_net = self.U[0](skip)
        L = L + skip + H_net * self.resnet_weight

        skip = self.convBlock[2](L)
        L_net = self.P[1](skip)
        H = H + skip + L_net * self.resnet_weight

        skip = self.convBlock[3](H)
        H_net = self.U[1](skip)
        L = L + skip + H_net * self.resnet_weight
        # scale
        if(self.scale==1):
            nh = lifting_coeff[4] + self.nh * 0.1
            nl = lifting_coeff[5] + self.nl * 0.1
            H = H * nh
            L = L * nl
        return L, H

    def lifting_forward_column(self, L, H):
        # first lifting step

        skip, L_net = self.P_2(L)
        H = H + skip + L_net * self.resnet_weight

        skip, H_net = self.U_2(H)
        L = L + skip + H_net * self.resnet_weight

        # n_h = lifting_coeff[4] + n_h * 0.1
        # H = H * n_h
        #
        # n_l = lifting_coeff[5] + n_l * 0.1
        # L = L * n_l

        # second lifting step

        skip, L_net = self.P_2(L)
        H = H + skip + L_net * self.resnet_weight

        skip, H_net = self.U_2(H)
        L = L + skip + H_net * self.resnet_weight

        return L, H

