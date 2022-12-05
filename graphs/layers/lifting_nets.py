import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as tdist
from graphs.layers.masked_conv2d import MaskedConv2d
from compressai.layers import GDN, GDN1
import sys


class SplitMergeSpatiallyForLiftingNet():
    """
    xxx
    """
    def __init__(self, split_mode='1p3'):
        """
        :param split_mode: split pixels spatially in 1+3 or 2+2 or 3+1 fashion : x_even (1,2,3 channels) and x_odd (3,2,1 channels)
        (Support for eve-odd row-column splitting is also added)
        """
        self.split_mode = split_mode
        assert self.split_mode in ('1p3', '2p2', '3p1', 'hor', 'ver')

    def split(self, x):
        """
        Splits tensor BxCxHxW spatially (i.e. using only H,W dimensions) into two tensors with quincux subsampling.
        The output trensors will have spatial dimension H/2xW/2 with twice the number of channels of input tensor.
        (Support for eve-odd row-column splitting is also added)
        :param x:   input image or tensor # B x C x H x W
        :return x_even, x_odd   2p2 : both    B x C*2 x H/2 x W/2
                                1p3 : x_even  B x C*1 x H/2 x W/2 , x_odd  B x C*3 x H/2 x W/2
                                3p1 : x_even  B x C*3 x H/2 x W/2 , x_odd  B x C*1 x H/2 x W/2
        """
        if x.shape[2] % 2 == 1 or x.shape[3] % 2 == 1:
            sys.exit(
                "split_tensor_spatially_before_lifting() rquires that tensor has even height and even width. Exiting.")
        # get the 4 sub pictures _ij from even/odd samples
        x_00 = x[:, :, 0::2, 0::2]
        x_01 = x[:, :, 0::2, 1::2]
        x_10 = x[:, :, 1::2, 0::2]
        x_11 = x[:, :, 1::2, 1::2]
        # form even odd outputs
        if self.split_mode == '2p2':  # i.e. 2 + 2
            x_even = torch.cat((x_00, x_11), dim=1)
            x_odd = torch.cat((x_01, x_10), dim=1)
        elif self.split_mode == '1p3':  # i.e. 1 + 3
            x_even = x_00
            x_odd = torch.cat((x_01, x_10, x_11), dim=1)
        elif self.split_mode == '3p1':  # i.e. 3 + 1
            x_even = torch.cat((x_00, x_01, x_10), dim=1)
            x_odd = x_11
        elif self.split_mode == 'hor':  # i.e. 1+1
            x_even = x[:, :, :, 0::2]   # B x C x H x W/2
            x_odd = x[:, :, :, 1::2]    # B x C x H x W/2
        elif self.split_mode == 'ver':  # i.e. 1+1
            x_even = x[:, :, 0::2, :]   # B x C x H/2 x W
            x_odd = x[:, :, 1::2, :]    # B x C x H/2 x W
        return x_even, x_odd

    def merge(self, x_even, x_odd):
        """
        Inverse operation of split_tensor_spatially_before_lifting(). Merges two tensors spatially (i.e. using
        only H,W dimensions) into a single tensor with quincux subsampling inverse. The output tensor will have
        twice spatial dimensions with half the number of channels (B x C x H x W) of input tensors.
        (Support for eve-odd row-column splitting is also added)
        :param x_even  see split definition for possible shapes
        :param x_odd   ses split ...
        :return x    # B x C x H x W
        """
        if x_even.shape[1] == x_odd.shape[1]:
            if not ((x_even.shape[1] % 2 == 0 and x_odd.shape[1] % 2 == 0) or (x_even.shape[1] % 2 == 1 and x_odd.shape[1] % 2 == 1)):
                sys.exit("merge_tensor_spatially_after_lifting() requires that tensors have even-or-odd "
                         "and equal channel dimension. Exiting.")
        else:
            if not (x_even.shape[1] * 3 == x_odd.shape[1] or x_even.shape[1] == x_odd.shape[1] * 3):
                sys.exit("merge_tensor_spatially_after_lifting() requires that x_odd tensor has 3 times "
                         "the channel dimension of x_even or veice versa. Exiting.")
        if self.split_mode == '2p2':
            x_00 = x_even[:, 0:x_even.shape[1] // 2, :, :]
            x_11 = x_even[:, x_even.shape[1] // 2:x_even.shape[1], :, :]
            x_01 = x_odd[:, 0:x_odd.shape[1] // 2, :, :]
            x_10 = x_odd[:, x_odd.shape[1] // 2:x_odd.shape[1], :, :]
            B, Cm2, Hd2, Wd2 = x_even.shape
            x = torch.empty(B, Cm2 // 2, Hd2 * 2, Wd2 * 2, device=x_even.device)
        elif self.split_mode == '1p3':
            x_00 = x_even
            x_01 = x_odd[:, 0:x_odd.shape[1] // 3, :, :]
            x_10 = x_odd[:, (x_odd.shape[1] // 3) * 1:(x_odd.shape[1] // 3) * 2, :, :]
            x_11 = x_odd[:, (x_odd.shape[1] // 3) * 2:(x_odd.shape[1] // 3) * 3, :, :]
            B, C, Hd2, Wd2 = x_even.shape
            x = torch.empty(B, C, Hd2 * 2, Wd2 * 2, device=x_even.device)
        elif self.split_mode == '3p1':
            x_11 = x_odd
            x_00 = x_even[:, 0:x_even.shape[1] // 3, :, :]
            x_01 = x_even[:, (x_even.shape[1] // 3) * 1:(x_even.shape[1] // 3) * 2, :, :]
            x_10 = x_even[:, (x_even.shape[1] // 3) * 2:(x_even.shape[1] // 3) * 3, :, :]
            B, C, Hd2, Wd2 = x_odd.shape
            x = torch.empty(B, C, Hd2 * 2, Wd2 * 2, device=x_even.device)
        elif self.split_mode == 'hor':
            B, C, H, Wd2 = x_odd.shape
            x = torch.empty(B, C, H, Wd2 * 2, device=x_even.device)
            x[:, :, :, 0::2] = x_even
            x[:, :, :, 1::2] = x_odd
            return x
        elif self.split_mode == 'ver':
            B, C, Hd2, W = x_odd.shape
            x = torch.empty(B, C, Hd2 * 2, W, device=x_even.device)
            x[:, :, 0::2, :] = x_even
            x[:, :, 1::2, :] = x_odd
            return x
        x[:, :, 0::2, 0::2] = x_00
        x[:, :, 0::2, 1::2] = x_01
        x[:, :, 1::2, 0::2] = x_10
        x[:, :, 1::2, 1::2] = x_11
        return x


def get_splitmode_se_so(config_split_mode):
    """ Function to get lifting in/out channels for each level from config_split_mode """
    if config_split_mode == 1:
        self_split_mode = '1p3'
        self_se, self_so = 1, 3
    elif config_split_mode == 3:
        self_split_mode = '3p1'
        self_se, self_so = 3, 1
    elif config_split_mode == 2:
        self_split_mode = '2p2'
        self_se, self_so = 2, 2
    elif config_split_mode == 'hv':
        self_split_mode = 'hor'
        self_se, self_so = 1, 3
    else:
        sys.exit('Unknown split mode. Exit.')
    return self_split_mode, self_se, self_so


class LiftingNet(nn.Module):
    """
    Performs multiple steps (num_lifting) of lifting, i.e. multiple update, prediction operations
    using inputs xi_even, xi_odd and produces output tensors xo_even, xo_odd.
    Note : Use this only with quincux split/merge. For ho+ver split/merge use LiftingHorVerNet
    """
    def __init__(self, in_xe_ch=6, in_xo_ch=6, out_xe_ch=6, out_xo_ch=6, precision_bits=0, num_lifting=1):
        """
        Constraints : out_xe_ch must be integer multiple of in_xe_ch
                      out_xo_ch must be integer multiple of in_xo_ch
        :param in_xe_ch, in_xo_ch, out_xe_ch, out_xo_ch are number of channels of inputs and outputs
        :param precision_bits is number of bits to use when rounding prediction and update results to have
        integer-to-integer mapping property in lifting
        """
        super(LiftingNet, self).__init__()
        if not (out_xe_ch % in_xe_ch == 0 and out_xo_ch % in_xo_ch == 0):
            sys.exit("out_xe_ch must be integer multiple of in_xe_ch. out_xe_ch must be integer multiple of in_xe_ch. Exiting.")
        self.precision_bits = precision_bits
        assert num_lifting in [1, 2, 3, 4, 5, 6]  # upto 6 successive lifting steps (i.e. prediction,update) are allowed
        self.num_lifting = num_lifting
        # self.RNDFACTOR = 2**precision_bits - 1
        self.RNDFACTOR = 255 * (2 ** (precision_bits - 8))
        self.in_xe_ch = in_xe_ch
        self.in_xo_ch = in_xo_ch
        self.out_xe_ch = out_xe_ch
        self.out_xo_ch = out_xo_ch
        self.pred_repeat = int(self.out_xo_ch / self.in_xo_ch)
        self.updt_repeat = int(self.out_xe_ch / self.in_xe_ch)
        # define some parameters of prediction and update NNs
        actfn = nn.Tanh()  # nn.LeakyReLU(negative_slope=0.1)  # nn.Tanh()
        ncnns = 2
        kersz = 3  # was 5
        # define first prediction and update NN.
        # first prediction and update may have number of output channels that are integer-multiples of numb input channs
        min_ch = min(self.in_xe_ch, self.out_xo_ch)
        max_ch = max(self.in_xe_ch, self.out_xo_ch)
        hid_ch = min(min_ch * 32, max(128, max_ch))  # was *16
        self.prediction = get_nn_sequential(self.in_xe_ch, hid_ch, self.out_xo_ch, actfn, num_cnns=ncnns, ker_size=kersz)
        min_ch = min(self.out_xo_ch, self.out_xe_ch)
        max_ch = max(self.out_xo_ch, self.out_xe_ch)
        hid_ch = min(min_ch * 32, max(128, max_ch))  # was *8
        self.update = get_nn_sequential(self.out_xo_ch, hid_ch, self.out_xe_ch, actfn, num_cnns=ncnns, ker_size=kersz)
        # if more than 1 prediction,update required, define additional pred and updt NN. These have same inp/out channls
        if self.num_lifting > 1:
            self.prediction2 = nn.ModuleList()
            min_ch = min(self.out_xe_ch, self.out_xo_ch)
            max_ch = max(self.out_xe_ch, self.out_xo_ch)
            hid_ch = min(min_ch * 32, max(128, max_ch))  # was *16
            for i in range(0, self.num_lifting-1, 1):
                self.prediction2.append(
                    get_nn_sequential(self.out_xe_ch, hid_ch, self.out_xo_ch, actfn, num_cnns=ncnns, ker_size=kersz)
                )
            self.update2 = nn.ModuleList()
            min_ch = min(self.out_xo_ch, self.out_xe_ch)
            max_ch = max(self.out_xo_ch, self.out_xe_ch)
            hid_ch = min(min_ch * 32, max(128, max_ch))   # was *8
            for i in range(0, self.num_lifting-1, 1):
                self.update2.append(
                    get_nn_sequential(self.out_xo_ch, hid_ch, self.out_xe_ch, actfn, num_cnns=ncnns, ker_size=kersz)
                )

    def round(self, x):
        if self.precision_bits > 0:
            if self.training:
                return x + (torch.rand_like(x) - 0.5) / self.RNDFACTOR
            else:
                return torch.round(x * self.RNDFACTOR) / self.RNDFACTOR
        else:
            return x

    def forward_lifting(self, in_xe, in_xo):
        out_xo = in_xo.repeat(1, self.pred_repeat, 1, 1) + self.round(self.prediction(in_xe))
        out_xe = in_xe.repeat(1, self.updt_repeat, 1, 1) + self.round(self.update(out_xo))
        if self.num_lifting > 1:
            for i in range(0, self.num_lifting-1, 1):
                out_xo = out_xo + self.round(self.prediction2[i](out_xe))
                out_xe = out_xe + self.round(self.update2[i](out_xo))
        return out_xe, out_xo

    def inverse_lifting(self, in_xe, in_xo):
        if self.num_lifting > 1:
            for i in range(self.num_lifting-2, -1, -1):
                in_xe = in_xe - self.round(self.update2[i](in_xo))
                in_xo = in_xo - self.round(self.prediction2[i](in_xe))
        out_xe = in_xe - self.round(self.update(in_xo))
        out_xe = out_xe[:, 0:self.in_xe_ch, :, :]  # unrepeat
        out_xo = in_xo - self.round(self.prediction(out_xe))
        out_xo = out_xo[:, 0:self.in_xo_ch, :, :]  # unrepeat
        return out_xe, out_xo


def get_nn_sequential(in_ch, hid_ch, out_ch, actfun=nn.Tanh(), num_cnns=3, ker_size=5):
    """ Return nn.sequential with desired number of layers, activation functions and channels """
    assert num_cnns > 0
    assert ker_size > 0
    layers = []
    # first layers upto (not including) final layer
    for i in range(0, num_cnns - 1):
        if i == 0:
            layers.append(nn.Conv2d(in_ch,  hid_ch, kernel_size=ker_size, stride=1, padding=ker_size//2, bias=False))
        else:
            layers.append(nn.Conv2d(hid_ch, hid_ch, kernel_size=ker_size, stride=1, padding=ker_size//2, bias=False))
        layers.append(actfun)
    # final layer
    if num_cnns > 1:
        layers.append(nn.Conv2d(hid_ch, out_ch, kernel_size=ker_size, stride=1, padding=ker_size//2, bias=False))
    else:
        layers.append(nn.Conv2d(in_ch,  out_ch, kernel_size=ker_size, stride=1, padding=ker_size//2, bias=False))
    # return nn seq with those layers
    return nn.Sequential(*layers)
    # e.g. : return nn.Sequential(
    #     nn.Conv2d(self.in_xe_ch, hid_ch,         5, stride=1, padding=5//2, bias=False), nn.Tanh(),
    #     nn.Conv2d(hid_ch,        hid_ch,         5, stride=1, padding=5//2, bias=False), nn.Tanh(),
    #     nn.Conv2d(hid_ch,        self.out_xo_ch, 5, stride=1, padding=5//2, bias=False)
    # )


class LiftingHorVerNet(nn.Module):
    """
    Performs multiple steps (num_lifting) of lifting, i.e. multiple update, prediction operations
    using inputs xi_even, xi_odd (which are hor split signals), then performs ver split and then another set of
    multiple steps (num_lifting) of lifting, i.e. multiple update, prediction operations, and produces
    output tensors xo_even, xo_odd, where xo_even contains the LL with ch dim C, xo_odd contains LH,HL,HH concatenated
    along channel dimension with ch dim C*3
    Note : Use this only with hor+ver split/merge. For quincux split/merge use LiftingHor
    """
    def __init__(self, in_xe_ch=6, in_xo_ch=6, out_xe_ch=6, out_xo_ch=6, precision_bits=0, num_lifting=1):
        """
        Constraints : out_xe_ch must be integer multiple of in_xe_ch
                      out_xo_ch must be integer multiple of in_xo_ch
        :param in_xe_ch, in_xo_ch, out_xe_ch, out_xo_ch are number of channels of inputs and outputs
        :param precision_bits is number of bits to use when rounding prediction and update results to have
        integer-to-integer mapping property in lifting
        """
        super(LiftingHorVerNet, self).__init__()
        self.lift_hor = LiftingNet(in_xe_ch=in_xe_ch, in_xo_ch=in_xo_ch,
                                   out_xe_ch=out_xe_ch, out_xo_ch=out_xo_ch,
                                   precision_bits=precision_bits, num_lifting=num_lifting)
        self.split_ver = SplitMergeSpatiallyForLiftingNet(split_mode='ver')
        # use two separate NN to lift/filter along  vertical direction the output of horizontal lifting
        self.lift_ver1 = LiftingNet(in_xe_ch=out_xe_ch, in_xo_ch=out_xo_ch,
                                    out_xe_ch=out_xe_ch, out_xo_ch=out_xo_ch,
                                    precision_bits=precision_bits, num_lifting=num_lifting)
        self.lift_ver2 = LiftingNet(in_xe_ch=out_xe_ch, in_xo_ch=out_xo_ch,
                                    out_xe_ch=out_xe_ch, out_xo_ch=out_xo_ch,
                                    precision_bits=precision_bits, num_lifting=num_lifting)

    def forward_lifting(self, in_xe, in_xo):
        out_xe, out_xo = self.lift_hor.forward_lifting(in_xe, in_xo)
        out_xe_spv_xe, out_xe_spv_xo = self.split_ver.split(out_xe)
        out_xo_spv_xe, out_xo_spv_xo = self.split_ver.split(out_xo)
        out_xe_xe, out_xe_xo = self.lift_ver1.forward_lifting(out_xe_spv_xe, out_xe_spv_xo)
        out_xo_xe, out_xo_xo = self.lift_ver2.forward_lifting(out_xo_spv_xe, out_xo_spv_xo)
        return out_xe_xe, torch.cat((out_xe_xo, out_xo_xe, out_xo_xo), dim=1)

    def inverse_lifting(self, in_xe, in_xo):
        out_xe_xe = in_xe
        out_xe_xo, out_xo_xe, out_xo_xo = in_xo.chunk(3, dim=1)
        out_xo_spv_xe, out_xo_spv_xo = self.lift_ver2.inverse_lifting(out_xo_xe, out_xo_xo)
        out_xe_spv_xe, out_xe_spv_xo = self.lift_ver1.inverse_lifting(out_xe_xe, out_xe_xo)
        out_xo = self.split_ver.merge(out_xo_spv_xe, out_xo_spv_xo)
        out_xe = self.split_ver.merge(out_xe_spv_xe, out_xe_spv_xo)
        in_xe, in_xo = self.lift_hor.inverse_lifting(out_xe, out_xo)
        return in_xe, in_xo
