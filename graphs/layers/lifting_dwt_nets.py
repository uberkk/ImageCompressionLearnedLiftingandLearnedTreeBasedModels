import numpy as np
import torch
from torch import nn
from graphs.layers.lifting_nets import LiftingNet, LiftingHorVerNet, SplitMergeSpatiallyForLiftingNet
import torch.nn.functional as F
from graphs.layers.wavelet_forward_v2 import wavelet_forward_v2
from graphs.layers.wavelet_inverse_v2 import wavelet_inverse_v2
from graphs.layers.P_block_v2 import P_block_v2
from graphs.layers.post_processing_networks import DnCNN, PostProcessingiWave, IRCNN, DIDN, DUDnCNN
import matplotlib.pyplot as plt
class DWTLayer(nn.Module):
    """ Multiple-level DWT with lifting (auto-encoder).
        Uses quincux split/merge which splits 2x2 pixels spatially in 1+3 or 2+2 or 3+1 fashion and puts them into channel dim:
        x : C x H x W  --->  x_even : (1/2/3)*C x H/2 x W/2
                             x_odd  : (3/2/1)*C x H/2 x W/2
    """
    def __init__(self, config):
        """
        Performs DWT with lifting, both encoder and decoder
        :param config:
        """
        super(DWTLayer, self).__init__()
        # set parameters and the auto-encoder network
        # precision bits of lifting steps P and U
        self.prec_bits = config.lif_prec_bits
        # number of lifting layers in DWT decomposition
        self.num_lifting_layers = config.dwtlevels
        self.num_lifting_perlayer = config.num_lifting_perlayer
        assert(self.num_lifting_layers > 0)  # at least 1 layer
        # how to split; split, merge class
        self.se, self.so = 1,3
        self.splitmerge = SplitMergeSpatiallyForLiftingNet(split_mode=self.split_mode)
        # color channels in input; define mulitple lifting layers
        self.clrch = config.clrch
        se, so = self.se * self.clrch, self.so * self.clrch
        self.lifting_layers = nn.ModuleList()
        for i in range(0, self.num_lifting_layers, 1):
            if self.split_mode == 'hor':   # i.e. hor+ver split mode
                self.lifting_layers.append(LiftingHorVerNet(in_xe_ch=se, in_xo_ch=se,
                                                            out_xe_ch=se, out_xo_ch=se,
                                                            precision_bits=self.prec_bits,
                                                            num_lifting=self.num_lifting_perlayer))
            else:  # quincux split modes
                self.lifting_layers.append(LiftingNet(in_xe_ch=se, in_xo_ch=so,
                                                      out_xe_ch=se, out_xo_ch=so,
                                                      precision_bits=self.prec_bits,
                                                      num_lifting=self.num_lifting_perlayer))
            so = se * self.so  # computation or der of this line must be before next line (bec o.w. se changes)
            se = se * self.se

    def encode(self, x):
        """
        :param x: input image to apply lifting based DWT (i.e. auto-encoders encode) B x C x H x W
        :return out_xe:
        :return out_xo_list:
        """
        # list to store out_xo layers from each lifting layer (will be quantized and sent to decoder)
        out_xo_list = []
        # apply split and lifting repeatedly (to xe )
        in_xe = x
        for i in range(0, self.num_lifting_layers, 1):
            x_even, x_odd = self.splitmerge.split(in_xe)
            out_xe, out_xo = self.lifting_layers[i].forward_lifting(x_even, x_odd)
            in_xe = out_xe
            out_xo_list.append(out_xo)
        return out_xe, out_xo_list

    def decode(self, out_xe, out_xo_list):
        """
        :param out_xe:
        :param out_xo_list:
        :return:
        """
        # apply inverse lifting and merge repeatedly
        for i in range(self.num_lifting_layers - 1, -1, -1):
            rec_xe, rec_xo = self.lifting_layers[i].inverse_lifting(out_xe, out_xo_list[i])
            out_xe = self.splitmerge.merge(rec_xe, rec_xo)
        return out_xe

from compressai.layers import GDN

class SubbandAutoEncoder(nn.Module):
    """ Auto-encoder layer for a subband of DWT2 layer.
        This autoencoder should use 1x1 kernels and groups=in_channels so that each subband coefficient is mapped by
        itself to anther value, i.e. coefs are indep and thier marginals are modified by ae to do rd-optimal
        quantization of the coefs in a subband.
    """
    def __init__(self, in_ch):
        """
        Performs Auto-encoders both encoder and decoder
        :param config:
        """
        super(SubbandAutoEncoder, self).__init__()
        # set parameters and the auto-encoder network
        K = 1
        P = K // 2
        iC = in_ch  # 3 * config.clrch  # number of subbands/channels in a layer of DWT2
        H = 32  # config.H
        self.ae_down = nn.Sequential(
            nn.Conv2d(iC * 1, iC * H, kernel_size=K, stride=1, padding=P, groups=iC), nn.Tanh(),  # GDN(iC * H),
            nn.Conv2d(iC * H, iC * H, kernel_size=K, stride=1, padding=P, groups=iC), nn.Tanh(),  # GDN(iC * H),
            nn.Conv2d(iC * H, iC * H, kernel_size=K, stride=1, padding=P, groups=iC), nn.Tanh(),  # GDN(iC * H),
            nn.Conv2d(iC * H, iC * 1, kernel_size=K, stride=1, padding=P, groups=iC)
        )
        self.ae_up = nn.Sequential(
            nn.ConvTranspose2d(iC * 1, iC * H, kernel_size=K, stride=1, padding=P, groups=iC, output_padding=1-1), nn.Tanh(),  # GDN(iC * H, inverse=True),
            nn.ConvTranspose2d(iC * H, iC * H, kernel_size=K, stride=1, padding=P, groups=iC, output_padding=1-1), nn.Tanh(),  # GDN(iC * H, inverse=True),
            nn.ConvTranspose2d(iC * H, iC * H, kernel_size=K, stride=1, padding=P, groups=iC, output_padding=1-1), nn.Tanh(),  # GDN(iC * H, inverse=True),
            nn.ConvTranspose2d(iC * H, iC * 1, kernel_size=K, stride=1, padding=P, groups=iC, output_padding=1-1)
        )

    def encode(self, x):
        """
        :param x:g
        :return:
        """
        return self.ae_down(x)

    def decode(self, y_hat):
        """
        :param y_hat:
        :return:
        """
        return self.ae_up(y_hat)
        # return torch.nan_to_num(self.ae_up(y_hat), nan=0.0, posinf=None, neginf=None)  # get rid of nan values
class SubbandAutoEncoderBerk(nn.Module):

    def __init__(self, in_ch):
        """
        Performs scaling before quantization and after quantization
        :param config:
        """
        super(SubbandAutoEncoderBerk, self).__init__()
        # set parameters and the auto-encoder network
        K = 3
        P = K // 2
        iC = in_ch
        H = 64  # config.H
        self.ae_down = nn.Sequential(
            nn.Conv2d(iC * 1, iC * H//2, kernel_size=K, stride=1, padding=P), GDN(iC * H//2),
            nn.Conv2d(iC * H//2, iC * H, kernel_size=K, stride=1, padding=P), GDN(iC * H),
            nn.Conv2d(iC * H, iC * H//2, kernel_size=K, stride=1, padding=P), GDN(iC * H//2),
            nn.Conv2d(iC * H//2, iC * 1, kernel_size=K, stride=1, padding=P)
        )
        self.ae_up = nn.Sequential(
            nn.ConvTranspose2d(iC * 1, iC * H//2, kernel_size=K, stride=1, padding=P), GDN(iC * H//2, inverse=True),
            nn.ConvTranspose2d(iC * H//2, iC * H, kernel_size=K, stride=1, padding=P), GDN(iC * H, inverse=True),
            nn.ConvTranspose2d(iC * H, iC * H//2, kernel_size=K, stride=1, padding=P), GDN(iC * H//2, inverse=True),
            nn.ConvTranspose2d(iC * H//2, iC * 1, kernel_size=K, stride=1, padding=P)
        )

    def encode(self, x):
        """
        :param x:g
        :return:
        """
        return self.ae_down(x)

    def decode(self, y_hat):
        """
        :param y_hat:
        :return:
        """
        return self.ae_up(y_hat)
        # return torch.nan_to_num(self.ae_up(y_hat), nan=0.0, posinf=None, neginf=None)  # get rid of nan values
class LinearSubbandAutoEncoder(nn.Module):
    """ Auto-encoder layer for a subband of DWT2 layer.
        This autoencoder should use 1x1 kernels and groups=in_channels so that each subband coefficient is mapped by
        itself to anther value, i.e. coefs are indep and thier marginals are modified by ae to do rd-optimal
        quantization of the coefs in a subband.
    """
    def __init__(self, in_ch):
        """
        Performs Auto-encoders both encoder and decoder
        :param config:
        """
        super(SubbandAutoEncoder, self).__init__()
        # set parameters and the auto-encoder network
        K = 1
        P = K // 2
        iC = in_ch  # 3 * config.clrch  # number of subbands/channels in a layer of DWT2
        H = 32  # config.H
        self.ae_down = nn.Sequential(
            nn.Conv2d(iC * 1, iC * H, kernel_size=K, stride=1, padding=P, groups=iC),   # GDN(iC * H),
            nn.Conv2d(iC * H, iC * H, kernel_size=K, stride=1, padding=P, groups=iC),   # GDN(iC * H),
            nn.Conv2d(iC * H, iC * H, kernel_size=K, stride=1, padding=P, groups=iC),  # GDN(iC * H),
            nn.Conv2d(iC * H, iC * 1, kernel_size=K, stride=1, padding=P, groups=iC)
        )
        self.ae_up = nn.Sequential(
            nn.ConvTranspose2d(iC * 1, iC * H, kernel_size=K, stride=1, padding=P, groups=iC, output_padding=1-1),  # GDN(iC * H, inverse=True),
            nn.ConvTranspose2d(iC * H, iC * H, kernel_size=K, stride=1, padding=P, groups=iC, output_padding=1-1),   # GDN(iC * H, inverse=True),
            nn.ConvTranspose2d(iC * H, iC * H, kernel_size=K, stride=1, padding=P, groups=iC, output_padding=1-1),  # GDN(iC * H, inverse=True),
            nn.ConvTranspose2d(iC * H, iC * 1, kernel_size=K, stride=1, padding=P, groups=iC, output_padding=1-1)
        )

    def encode(self, x):
        """
        :param x:g
        :return:
        """
        return self.ae_down(x)

    def decode(self, y_hat):
        """
        :param y_hat:
        :return:
        """
        return self.ae_up(y_hat)
        # return torch.nan_to_num(self.ae_up(y_hat), nan=0.0, posinf=None, neginf=None)  # get rid of nan values

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
class DWTPytorchWaveletsLayer(nn.Module):
    """ Multiple-level DWT with JPEG2000 9/7 filters (auto-encoder).
        ... describe
    """
    def __init__(self, config):
        """
        :param config:
        """
        super(DWTPytorchWaveletsLayer, self).__init__()
        # set parameters and the auto-encoder network
        # number of decompositions layers in DWT decomposition
        self.dwtlevels = config.dwtlevels
        # color channels in input; define mulitple lifting layers
        self.clrch = config.clrch
        self.conv_grps = config.clrch
        # get DWT objects from pytorch_wavelets
        extmod = 'periodization'
        wtname = 'bior4.4'
        self.xfm = DWTForward(J=self.dwtlevels, mode=extmod, wave=wtname)  # Accepts all wave types available to PyWavelets
        self.ifm = DWTInverse(mode=extmod, wave=wtname)
        # scaling parameter
        #self.scl_ = nn.Parameter(nn.init.constant_(torch.empty(1, 1, 1, 1), 9.0), requires_grad=False)  # scaling params
        #self.scl_[0, 1:3, 0, 0] = 2.5  # chroma/GB channels have larger quantization
        # auto-encoders to perform rd-optimal quantization of subbands
        self.Yl_ae = SubbandAutoEncoder(in_ch=1 * config.clrch)
        self.Yh_ae = nn.ModuleList()
        for i in range(0, self.dwtlevels):
            self.Yh_ae.append(SubbandAutoEncoder(in_ch=3 * config.clrch))

    def encode(self, x):
        """
        :param x: input image to apply DWT (i.e. auto-encoders encode) B x C x H x W
        :return out_xe:
        :return out_xo_list:
        """
        # scale input image
        #x = x * self.scl_.expand_as(x)
        # apply DWT2
        Yl, Yh = self.xfm(x)
        # list to store out_xo layers from each lifting layer (will be quantized and sent to decoder)
        out_xe = self.Yl_ae.encode(Yl)  # Yl
        out_xo_list = []
        for i in range(0, self.dwtlevels):
            B, C, Three, H, W = Yh[i].shape
            Yh[i] = Yh[i].view(B, C*3, H, W)
            Yh_ae = self.Yh_ae[i].encode(Yh[i])
            out_xo_list.append(Yh_ae)
        return out_xe, out_xo_list

    def decode(self, out_xe, out_xo_list):
        """
        :param out_xe:
        :param out_xo_list:
        :return:
        """
        Yl = self.Yl_ae.decode(out_xe)  # out_xe
        Yh = []
        # apply inverse lifting and merge repeatedly
        for i in range(0, self.dwtlevels):
            Yh_ae = self.Yh_ae[i].decode(out_xo_list[i])
            B, C, H, W = out_xo_list[i].shape
            Yh.append(Yh_ae.view(B, C//3, 3, H, W))
        xhat = self.ifm((Yl, Yh))
        # ivnerse scale reconstruction image
        #xhat = xhat / self.scl_.expand_as(xhat)
        return xhat
from graphs.layers import cbam

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out
class PostProcessing(nn.Module):
    def __init__(self, config):
        super(PostProcessing, self).__init__()
        self.attentionInputChannel = 64
        self.res_connection_weight = nn.Parameter(torch.tensor(0.1))
        self.kernelSize = 5
        self.beforeAttentionCNN = nn.Conv2d(config.clrch, self.attentionInputChannel * config.clrch,kernel_size=self.kernelSize
                      ,padding = self.kernelSize//2)
        self.attentionBlock = cbam.CBAM(self.attentionInputChannel* config.clrch,16)
        self.afterAttentionCNN = nn.Sequential(nn.Conv2d(self.attentionInputChannel*config.clrch,
                                                         int(self.attentionInputChannel*config.clrch/8), kernel_size=self.kernelSize
                                                            ,padding=self.kernelSize//2),
            nn.Tanh(),
            nn.Conv2d(int(self.attentionInputChannel*config.clrch/8), int((self.attentionInputChannel*config.clrch/8)/(self.attentionInputChannel/8)),
                      kernel_size=self.kernelSize, padding=self.kernelSize//2))
    def forward(self, inputImage):
        beforeCBAM = self.beforeAttentionCNN(inputImage)
        afterCBAM = self.attentionBlock(beforeCBAM)
        endOfAttention = self.afterAttentionCNN(afterCBAM)
        reconImage = endOfAttention + inputImage*self.res_connection_weight
        return reconImage
class DWTCDF97Layer(nn.Module):
    """ Multiple-level DWT with JPEG2000 9/7 filters (auto-encoder).
        ... describe
    """
    def __init__(self, config):
        """
        :param config:
        """
        super(DWTCDF97Layer, self).__init__()
        # set parameters and the auto-encoder network
        # number of decompositions layers in DWT decomposition
        self.dwtlevels = 1 # config.dwtlevels
        # color channels in input; define mulitple lifting layers
        self.clrch = 1  # config.clrch
        self.conv_grps = 1  # config.clrch
        # get CDF 9/7 filters in 2D
        h_ana_LL, h_ana_LH, h_ana_HL, h_ana_HH, h_syn_LL, h_syn_LH, h_syn_HL, h_syn_HH = \
            get_cdf97_filters(oned_or_twod='2D')
        # torch.nn.functional.conv2d.html : weight – filters of shape (out_channels, in_channels/groups, kH, kW)
        self.analysis_weights = torch.empty(4, 3//3, 9, 1+7+1)
        self.analysis_weights = torch.stack((h_ana_LL, h_ana_LH, h_ana_HL, h_ana_HH)).unsqueeze(dim=1)
        self.synthesi_weights = torch.empty(4, 3//3, 9, 1+7+1)
        self.synthesi_weights = torch.stack((h_syn_LL, h_syn_LH, h_syn_HL, h_syn_HH)).unsqueeze(dim=1)
        if self.clrch == 3:
            self.analysis_weights = self.analysis_weights.repeat(3, 1, 1, 1)
            self.synthesi_weights = self.synthesi_weights.repeat(3, 1, 1, 1)
        # make them nn.Par to get them to cuda when .to()
        self.analysis_weights = nn.Parameter(self.analysis_weights, requires_grad=False)
        self.synthesi_weights = nn.Parameter(self.synthesi_weights, requires_grad=False)
        # get padding object
        self.Hpad, self.Wpad = self.analysis_weights.shape[2]//1, self.analysis_weights.shape[3]//1
        self.replicPad2d = nn.ReplicationPad2d((self.Wpad, self.Wpad, self.Hpad, self.Hpad))

    def encode(self, x):
        """
        :param x: input image to apply DWT (i.e. auto-encoders encode) B x C x H x W
        :return out_xe:
        :return out_xo_list:
        """
        # list to store out_xo layers from each lifting layer (will be quantized and sent to decoder)
        out_xo_list = []
        # apply 1 level decomposition repeatedly (to xe )
        in_xe = x
        for i in range(0, self.dwtlevels, 1):
            # in_xe = self.replicPad2d(in_xe)
            out_LL_LH_HL_HH = F.conv2d(in_xe, self.analysis_weights, bias=None, stride=2, padding=9, groups=self.conv_grps)
            out_xe = out_LL_LH_HL_HH[:, 0::4, :, :]
            out_xo = torch.cat((out_LL_LH_HL_HH[:, 1::4, :, :], out_LL_LH_HL_HH[:, 2::4, :, :], out_LL_LH_HL_HH[:, 3::4, :, :]), dim=1)
            in_xe = out_xe
            out_xo_list.append(out_xo)
        return out_xe, out_xo_list

    def decode(self, out_xe, out_xo_list):
        """
        :param out_xe:
        :param out_xo_list:
        :return:
        """
        # apply inverse lifting and merge repeatedly
        for i in range(self.dwtlevels - 1, -1, -1):
            B, Cxe, H, W = out_xe.shape
            _, Cxo, _, _ = out_xo_list[i].shape
            out_LL_LH_HL_HH = torch.empty(B, Cxe+Cxo, H, W, device=out_xe.device).double()
            out_LL_LH_HL_HH[:, 0::4, :, :] = out_xe
            out_LL_LH_HL_HH[:, 1::4, :, :] = out_xo_list[i][:, 0 * self.clrch:1 * self.clrch, :, :]
            out_LL_LH_HL_HH[:, 2::4, :, :] = out_xo_list[i][:, 1 * self.clrch:2 * self.clrch, :, :]
            out_LL_LH_HL_HH[:, 3::4, :, :] = out_xo_list[i][:, 2 * self.clrch:3 * self.clrch, :, :]
            # out_LL_LH_HL_HH = self.replicPad2d(out_LL_LH_HL_HH)
            out_xe = F.conv_transpose2d(out_LL_LH_HL_HH, self.synthesi_weights, bias=None, stride=2,
                                        padding=8, groups=self.conv_grps, output_padding=0)
        #return out_xe[:, :, 2*self.Hpad-1:-2*self.Hpad+1, 2*self.Wpad-1:-2*self.Wpad+1]  # off=8; err = torch.abs(x - xhat[0:1, 0:1, off:off+100, off:off+100]); err[0, 0, 8:92, 9:91].sum()
        q = 0; print(f'q={q}')
        return out_xe[:, :, 0:-2, 0:-2]

def show_wavelet_coeff(out_xe, out_xo_list, shape, config):
    coeff_array = torch.empty(size=(shape))
    B,C, H, W = out_xe.cpu().shape
    # out_xe = out_xe.cuda()
    out_xe = out_xe
    coeff_array[:,:,0:H, 0:W] = out_xe
    out_xo_list.reverse()
    for i in range(config.dwtlevels):
        coeff_array[:, :, 0:H * np.power(2, i ), W * np.power(2, i):W * np.power(2, i+1)] = out_xo_list[i][:, 0:1, :, :].detach()
        coeff_array[:, :, (np.power(2, i)) * H:H * np.power(2, i+1), 0:W*np.power(2, i)] = out_xo_list[i][:, 1:2, :, :].detach()
        coeff_array[:, :, (np.power(2, i)) * H:H * np.power(2, i + 1), W * np.power(2, i):W * np.power(2, i+1)] = out_xo_list[i][:, 2:3, :, :].detach()
    out_xo_list.reverse()
    coeff_array_numpy = coeff_array.cpu()
    coeff_array_numpy = (coeff_array_numpy.squeeze(0)).squeeze(0)
    plt.figure()
    plt.imshow(coeff_array_numpy, cmap='gray')
    plt.show()
    return None



def get_cdf97_filters(oned_or_twod='2D'):
    h_ana_lp = torch.tensor([0.0,  0.037828455507264, -0.023849465019557, -0.110624404418437,  0.377402855612831,  0.852698679008894,  0.377402855612831, -0.110624404418437, -0.023849465019557,  0.037828455507264], dtype=torch.double)
    h_ana_hp = torch.tensor([0.0, -0.064538882628697,  0.040689417609164,  0.418092273221617, -0.788485616405583,  0.418092273221617,  0.040689417609164, -0.064538882628697,  0.0,                0.0],               dtype=torch.double)
    h_syn_lp = torch.tensor([0.0, -0.064538882628697, -0.040689417609164,  0.418092273221617,  0.788485616405583,  0.418092273221617, -0.040689417609164, -0.064538882628697,  0.0,                0.0],               dtype=torch.double)
    h_syn_hp = torch.tensor([0.0, -0.037828455507264, -0.023849465019557,  0.110624404418437,  0.377402855612831, -0.852698679008894,  0.377402855612831,  0.110624404418437, -0.023849465019557, -0.037828455507264], dtype=torch.double)
    h_ana_LL = h_ana_lp.view(-1, 1) * h_ana_lp.view(1, -1)
    h_ana_LH = h_ana_lp.view(-1, 1) * h_ana_hp.view(1, -1)
    h_ana_HL = h_ana_hp.view(-1, 1) * h_ana_lp.view(1, -1)
    h_ana_HH = h_ana_hp.view(-1, 1) * h_ana_hp.view(1, -1)
    h_syn_LL = h_syn_lp.view(-1, 1) * h_syn_lp.view(1, -1)
    h_syn_LH = h_syn_lp.view(-1, 1) * h_syn_hp.view(1, -1)
    h_syn_HL = h_syn_hp.view(-1, 1) * h_syn_lp.view(1, -1)
    h_syn_HH = h_syn_hp.view(-1, 1) * h_syn_hp.view(1, -1)
    if oned_or_twod == '1D':
        return h_ana_lp, h_ana_hp, h_syn_lp, h_syn_hp
    elif oned_or_twod == '2D':
        return h_ana_LL, h_ana_LH, h_ana_HL, h_ana_HH, h_syn_LL, h_syn_LH, h_syn_HL, h_syn_HH
lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781,
                 1.149604398860241]  # bior4.4
class BasicWavelet(nn.Module):
    def __init__(self,config):
        super(BasicWavelet, self).__init__()
        self.waveletLevel = config.dwtlevels
        self.liftingLevel = config.num_lifting_perlayer
        self.blockprop = config.block_property
        self.clrch = config.clrch
        self.linearityflag = config.linearity_flag
        self.conv_filter_size = 3

        self.forwardTransform = nn.ModuleList()
        self.inverseTransform = nn.ModuleList()
        # self.preProcessingList = preProcessBlock(config.clrch, config.filtersize)
        self.liftingConvForward = nn.ModuleList()
        self.liftingConvInverse = nn.ModuleList()
        # self.Yl_ae = SubbandAutoEncoder(in_ch=1 * config.clrch)
        self.Yh_ae = nn.ModuleList()
        if(config.autoencoder == "SubbandAutoEncoder"):
            self.Yl_ae = SubbandAutoEncoder(in_ch=1 * config.clrch)
            for i in range(0, self.waveletLevel):
                self.Yh_ae.append(SubbandAutoEncoder(in_ch=3 * config.clrch))
        elif(config.autoencoder == "SubbandAutoEncoderBerk"):
            self.Yl_ae = SubbandAutoEncoderBerk(in_ch=1 * config.clrch)
            for i in range(0, self.waveletLevel):
                self.Yh_ae.append(SubbandAutoEncoderBerk(in_ch=3 * config.clrch))

        if (self.linearityflag==0):
            for waveletLevelindex in range(self.waveletLevel):
                self.forwardTransform.append(nn.Conv2d(self.clrch, self.clrch*4, kernel_size=self.conv_filter_size,
                                                 stride=2,padding = self.conv_filter_size//2))
                self.inverseTransform.append(nn.ConvTranspose2d(in_channels=self.clrch*4, out_channels=self.clrch,
                                                 kernel_size=self.conv_filter_size, stride=2,
                                                 padding = int((self.conv_filter_size-1)/2), output_padding=1))
                # for liftingLevelIndex in range(self.liftingLevel):
                #     self.liftingConvForward.append(nn.Conv2d(self.clrch, self.clrch, filter=self.conv_filter_size,
                #                                  stride=1, padding= self.conv_filter_size//2))
                #     self.liftingConvInverse.append(nn.Conv2d(self.clrch, self.clrch, filter=self.conv_filter_size,
                #                                  stride=1, padding= self.conv_filter_size//2))
        elif(self.linearityflag==1):
            for waveletLevelindex in range(self.waveletLevel):
                self.forwardTransform.append(nn.Sequential(nn.Conv2d(self.clrch, self.clrch*4, kernel_size=self.conv_filter_size,
                                                 stride=2,padding = self.conv_filter_size//2), GDN(self.clrch*4),
                                                nn.Conv2d(self.clrch*4, self.clrch * 4, kernel_size=self.conv_filter_size,
                                                                     stride=1, padding=self.conv_filter_size // 2)
                                                           ))
                self.inverseTransform.append(nn.Sequential(nn.Conv2d(in_channels=self.clrch*4, out_channels=self.clrch*4,
                                                 kernel_size=self.conv_filter_size, stride=1,
                                                 padding = self.conv_filter_size//2), GDN(self.clrch*4, inverse= True),
                                             nn.ConvTranspose2d(in_channels=self.clrch * 4, out_channels=self.clrch,
                                                                              kernel_size=self.conv_filter_size,
                                                                              stride=2,
                                                                              padding=int(
                                                                                  (self.conv_filter_size - 1) / 2),
                                                                              output_padding=1)))

    def encode(self, input):
        LL = input
        LL_list = []
        Yh_list = []
        LL_list.append(LL)
        for waveletLevelIndex in range(self.waveletLevel):
            downSampled = self.forwardTransform[waveletLevelIndex](LL)
            LL = downSampled[:,0:self.clrch,:,:]
            Yh = downSampled[:,self.clrch:,:,:].unsqueeze(1)
            Yh_list.append(Yh)
            LL_list.append(LL)
## 4x3x32x32 -> LL: 4x3x32x32 , H : 4x3x3x32x32
    #4x12x32x32 -> LL: 4x3x32x32,H: 4x3x128x128 -> 4x1x3x128x128 -> 4x3x128x128
        Yl =LL

        out_xe = self.Yl_ae.encode(Yl)  # Yl
        out_xo_list = []
        for i in range(0, self.waveletLevel):
            B, C, Three, H, W = Yh_list[i].shape
            Yh_list[i] = Yh_list[i].view(B, C*3, H, W)
            Yh_ae = self.Yh_ae[i].encode(Yh_list[i])
            out_xo_list.append(Yh_ae)
        return out_xe, out_xo_list

    def decode(self, out_xe, out_xo_list):
        Yl = self.Yl_ae.decode(out_xe)  # out_xe
        Yh = []
        # apply inverse lifting and merge repeatedly
        for i in range(0, self.waveletLevel):
            Yh_ae = self.Yh_ae[i].decode(out_xo_list[i])
            # B, C, H, W = out_xo_list[i].shape
            # Yh.append(Yh_ae.view(B, C//3, 3, H, W))
            Yh.append(Yh_ae)
        LL = Yl
        for waveletLevelIndex in range(self.waveletLevel):
            concatArray = torch.cat((LL,Yh[self.waveletLevel-waveletLevelIndex-1]),1)
            LL = self.inverseTransform[waveletLevelIndex](concatArray)

        return LL
class AttentionWavelet(nn.Module):
    '''
    This module is used in networks that require a shortcut.
    X --> output, LL(shortcut)
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, config):
        super(AttentionWavelet, self).__init__()
        extmod = 'periodization'
        wtname = 'bior4.4'
        self.dwtlevels = config.dwtlevels
        self.dwt = DWTForward(J=self.dwtlevels, mode=extmod, wave=wtname)  # Accepts all wave types available to PyWavelets
        self.idwt = DWTInverse(mode=extmod, wave=wtname)
        self.softmax = nn.Softmax2d()
        self.Yl_ae = SubbandAutoEncoder(in_ch=1 * config.clrch)
        self.Yh_ae = nn.ModuleList()
        for i in range(0, self.dwtlevels):
            self.Yh_ae.append(SubbandAutoEncoder(in_ch=3 * config.clrch))


    @staticmethod
    def get_module_name():
        return "wa"

    def encode(self, input):
        LL, Yh_list = self.dwt(input)
        output = LL

        LH = Yh_list[self.dwtlevels-1][:,:,0,:]
        HL = Yh_list[self.dwtlevels-1][:,:,1,:]

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        LL = output
        out_xe = self.Yl_ae.encode(LL)  # Yl
        out_xo_list = []

        for i in range(0, self.dwtlevels):
            B, C, Three, H, W = Yh_list[i].shape
            Yh_list[i] = Yh_list[i].view(B, C*3, H, W)
            Yh_ae = self.Yh_ae[i].encode(Yh_list[i])
            out_xo_list.append(Yh_ae)

        return out_xe, out_xo_list

    def decode(self, out_xe, out_xo_list):
        Yl = self.Yl_ae.decode(out_xe)  # out_xe
        Yh = []
        # apply inverse lifting and merge repeatedly
        for i in range(0, self.dwtlevels):
            Yh_ae = self.Yh_ae[i].decode(out_xo_list[i])
            B, C, H, W = out_xo_list[i].shape
            Yh.append(Yh_ae.view(B, C//3, 3, H, W))
        xhat = self.idwt((Yl, Yh))

        return xhat
class AttentionWaveletPostProcessing(nn.Module):
    '''
    This module is used in networks that require a shortcut.
    X --> output, LL(shortcut)
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, config):
        super(AttentionWaveletPostProcessing, self).__init__()
        extmod = 'periodization'
        wtname = 'bior4.4'
        self.dwtlevels = config.dwtlevels
        self.dwt = DWTForward(J=self.dwtlevels, mode=extmod, wave=wtname)  # Accepts all wave types available to PyWavelets
        self.idwt = DWTInverse(mode=extmod, wave=wtname)
        self.softmax = nn.Softmax2d()
        self.Yl_ae = SubbandAutoEncoder(in_ch=1 * config.clrch)
        self.Yh_ae = nn.ModuleList()
        for i in range(0, self.dwtlevels):
            self.Yh_ae.append(SubbandAutoEncoder(in_ch=3 * config.clrch))

        self.postprocess = PostProcessing(config)


    @staticmethod
    def get_module_name():
        return "wa"

    def encode(self, input):
        LL, Yh_list = self.dwt(input)
        output = LL

        LH = Yh_list[self.dwtlevels-1][:,:,0,:]
        HL = Yh_list[self.dwtlevels-1][:,:,1,:]

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        LL = output
        out_xe = self.Yl_ae.encode(LL)  # Yl
        out_xo_list = []

        for i in range(0, self.dwtlevels):
            B, C, Three, H, W = Yh_list[i].shape
            Yh_list[i] = Yh_list[i].view(B, C*3, H, W)
            Yh_ae = self.Yh_ae[i].encode(Yh_list[i])
            out_xo_list.append(Yh_ae)

        return out_xe, out_xo_list

    def decode(self, out_xe, out_xo_list):
        Yl = self.Yl_ae.decode(out_xe)  # out_xe
        Yh = []
        # apply inverse lifting and merge repeatedly
        for i in range(0, self.dwtlevels):
            Yh_ae = self.Yh_ae[i].decode(out_xo_list[i])
            B, C, H, W = out_xo_list[i].shape
            Yh.append(Yh_ae.view(B, C//3, 3, H, W))
        xhat = self.idwt((Yl, Yh))
        output_image = self.postprocess(xhat)

        return output_image
class LiftingBasedNeuralWaveletv4(nn.Module):
    def __init__(self, config):
        super(LiftingBasedNeuralWaveletv4, self).__init__()
        self.waveletLevel = config.dwtlevels
        self.liftingLevel = config.num_lifting_perlayer
        self.blockprop = config.block_property
        self.clrch = config.clrch
        self.linearityflag = config.linearity_flag
        self.conv_filter_size = config.filtersize
        self.postprocessflag = config.postprocess
        self.res_connection_weight = config.res_connection_weight
        self.P_blocks = nn.ModuleList()
        self.U_blocks = nn.ModuleList()
        self.waveletForward = nn.ModuleList()
        self.waveletInverse = nn.ModuleList()
        self.Yh_ae = nn.ModuleList()
        self.config = config
        self.depth_scale = config.depth_scale * 8
        # self.postprocessH = nn.ModuleList()
        self.preProcessingList = self.preProcessBlock(config.clrch, config.filtersize)
        if(config.autoencoder == "SubbandAutoEncoder"):
            self.Yl_ae = SubbandAutoEncoder(in_ch=1 * config.clrch)
            for i in range(0, self.waveletLevel):
                self.Yh_ae.append(SubbandAutoEncoder(in_ch=3 * config.clrch))
        elif(config.autoencoder == "SubbandAutoEncoderBerk"):
            self.Yl_ae = SubbandAutoEncoderBerk(in_ch=1 * config.clrch)
            for i in range(0, self.waveletLevel):
                self.Yh_ae.append(SubbandAutoEncoderBerk(in_ch=3 * config.clrch))
        # if (self.postprocessflag == 'DnCNN'):
        #     self.postprocess = DnCNN(self.clrch)
        # elif (self.postprocessflag == 'iwave' ):
        #     self.postprocess = PostProcessingiWave(config)
        # elif (self.postprocessflag == 'IRCNN'):
        #     self.postprocess = IRCNN(self.clrch, self.clrch)
        # elif (self.postprocessflag == 'DIDN'):
        #     self.postprocess = DIDN(self.config)
        # elif (self.postprocessflag == 'DUDnCNN'):
        #     self.postprocess = DUDnCNN(self.config)
        # if (self.postprocessflag == 'x' ):
        #     self.postprocessL = PostProcessingiWave(config)
        #     for i in range(self.waveletLevel):
        #         self.postprocessH.append(PostProcessingiWave(config))


        if (self.blockprop == 'same'):
            numberOfBlocks = self.liftingLevel
            self.nh =nn.Parameter(nn.init.constant_(torch.empty(1, 1, 1, 1), 0.0), requires_grad=True)
            self.nl =nn.Parameter(nn.init.constant_(torch.empty(1, 1, 1, 1), 0.0), requires_grad=True)
        elif (self.blockprop == 'different'):
            numberOfBlocks = self.liftingLevel * 2 * self.waveletLevel
            self.nh =nn.Parameter(nn.init.constant_(torch.empty(1, 1, 1, 1), 0.0), requires_grad=True)
            self.nl =nn.Parameter(nn.init.constant_(torch.empty(1, 1, 1, 1), 0.0), requires_grad=True)

        for liftingLevelIndex in range(numberOfBlocks):
            self.P_blocks.append(P_block_v2(self.linearityflag, self.clrch, self.conv_filter_size,self.depth_scale))
            self.U_blocks.append(P_block_v2(self.linearityflag, self.clrch, self.conv_filter_size,self.depth_scale))

        if (self.blockprop == 'same'):
            for waveletLevelIndex in range(self.waveletLevel):
                self.waveletForward.append(
                    wavelet_forward_v2(self.P_blocks, self.U_blocks, self.res_connection_weight, self.liftingLevel,self.preProcessingList,
                                    self.config,self.nh,self.nl))
                self.waveletInverse.append(
                    wavelet_inverse_v2(self.P_blocks, self.U_blocks, self.res_connection_weight, self.liftingLevel,self.preProcessingList,
                                    self.config,self.nh,self.nl))
        elif (self.blockprop == 'different'):
            for waveletLevelIndex in range(self.waveletLevel):
                self.waveletForward.append(wavelet_forward_v2(
                    self.P_blocks[waveletLevelIndex * self.liftingLevel:(waveletLevelIndex + 1) * self.liftingLevel],
                    self.U_blocks[waveletLevelIndex * self.liftingLevel:(waveletLevelIndex + 1) * self.liftingLevel],
                    self.res_connection_weight, self.liftingLevel,self.preProcessingList,
                                    self.config,self.nh,self.nl))
                self.waveletInverse.append(wavelet_inverse_v2(
                    self.P_blocks[self.waveletLevel * self.liftingLevel:self.waveletLevel * self.liftingLevel + (waveletLevelIndex + 1) * self.liftingLevel],
                    self.U_blocks[self.waveletLevel * self.liftingLevel:self.waveletLevel * self.liftingLevel + (waveletLevelIndex + 1) * self.liftingLevel],
                    self.res_connection_weight, self.liftingLevel,self.preProcessingList,
                                    self.config,self.nh,self.nl))

    def encode(self, input):
        Yh = []
        LL_list = []
        LL_list.append(input)
        for waveletLevel in range(self.waveletLevel):
            LL, LH, HL, HH = self.waveletForward[waveletLevel].one_level_lifting(LL_list[waveletLevel])
            LL_list.append(LL)
            LH, HL, HH = LH.unsqueeze(2), HL.unsqueeze(2), HH.unsqueeze(2)
            Yh.append(torch.cat((LH, HL, HH), 2))
        # Yl = LL.unsqueeze(1)
        Yl = LL

        out_xe = self.Yl_ae.encode(Yl)  # Yl
        out_xo_list = []
        for i in range(0, self.waveletLevel):
            B, C, Three, H, W = Yh[i].shape
            Yh[i] = Yh[i].view(B, C * 3, H, W)
            Yh_ae = self.Yh_ae[i].encode(Yh[i])
            out_xo_list.append(Yh_ae)

        if ((self.config.mode == "validate" or self.config.mode == "test") and self.config.imshow_validation=="true"):
            show_wavelet_coeff(out_xe, out_xo_list, input.detach().cpu().shape, self.config)
        return out_xe, out_xo_list

    def decode(self, out_xe, out_xo_list):
        if self.config.mode != "test":
            Yl = self.Yl_ae.decode(out_xe.cuda())  # out_xe
        else:
            Yl = self.Yl_ae.decode(out_xe)  # out_xe
        Yh = []
        # apply inverse lifting and merge repeatedly
        for i in range(0, self.waveletLevel):
            if self.config.mode != "test":
                Yh_ae = self.Yh_ae[i].decode(out_xo_list[i].cuda())
            else:
                Yh_ae = self.Yh_ae[i].decode(out_xo_list[i])
            B, C, H, W = out_xo_list[i].shape
            Yh.append(Yh_ae.view(B, C // 3, 3, H, W))
        LL_list = []
        LL_list.append(Yl)
        for waveletLevel in range(self.waveletLevel):
            # Yh[self.liftingLevel-waveletLevel-1] = torch.cat((Yh[self.liftingLevel-waveletLevel-1], LL_list[waveletLevel]),2)
            LL = self.waveletInverse[self.waveletLevel - waveletLevel - 1].one_level_lifting(LL_list[waveletLevel],
                                                                           Yh[self.waveletLevel - waveletLevel - 1][:,
                                                                           :, 0, :, :],
                                                                           Yh[self.waveletLevel - waveletLevel - 1][:,
                                                                           :, 1, :, :],
                                                                           Yh[self.waveletLevel - waveletLevel - 1][:,
                                                                           :, 2, :, :],
                                                                           )
            LL_list.append(LL)

        # if (self.postprocessflag == 'DnCNN' or self.postprocessflag == 'iwave' or self.postprocessflag == 'IRCNN'):
        # if (self.postprocessflag != "none"):
        #     recon_image = self.postprocess(LL)
        # else:
        #     recon_image = LL
        recon_image = LL
        return recon_image

    def preProcessBlock(self, csize, conv_filter_size):
        conv1_params = torch.tensor(([0.0],
                                     [lifting_coeff[0]],
                                     [lifting_coeff[0]]))
        conv1_params = conv1_params.view(1, 1, 3, 1)

        conv2_params = torch.tensor(([lifting_coeff[1]],
                                     [lifting_coeff[1]],
                                     [0.0]))
        conv2_params = conv2_params.view(1, 1, 3, 1)

        conv3_params = torch.tensor(([0.0],
                                     [lifting_coeff[2]],
                                     [lifting_coeff[2]]))
        conv3_params = conv3_params.view(1, 1, 3, 1)

        conv4_params = torch.tensor(([lifting_coeff[3]],
                                     [lifting_coeff[3]],
                                     [0.0]))
        conv4_params = conv4_params.view(1, 1, 3, 1)

        conv1 = nn.Conv2d(1 * csize, 1 * csize, kernel_size=(3, 1),
                          stride=1, padding=(3 // 2, 0), bias=False)
        conv1.weight = torch.nn.Parameter(conv1_params, requires_grad=True)

        conv2 = nn.Conv2d(1 * csize, 1 * csize, kernel_size=(3, 1),
                          stride=1, padding=(3 // 2, 0), bias=False)
        conv2.weight = torch.nn.Parameter(conv2_params, requires_grad=True)

        conv3 = nn.Conv2d(1 * csize, 1 * csize, kernel_size=(3, 1),
                          stride=1, padding=(3 // 2, 0), bias=False)
        conv3.weight = torch.nn.Parameter(conv3_params, requires_grad=True)

        conv4 = nn.Conv2d(1 * csize, 1 * csize, kernel_size=(3, 1),
                          stride=1, padding=(3 // 2, 0), bias=False)
        conv4.weight = torch.nn.Parameter(conv4_params, requires_grad=True)

        convList = nn.ModuleList()
        convList.append(conv1)
        convList.append(conv2)
        convList.append(conv3)
        convList.append(conv4)

        return convList









