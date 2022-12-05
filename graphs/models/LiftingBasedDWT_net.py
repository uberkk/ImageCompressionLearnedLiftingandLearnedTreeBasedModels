import torch
from torch import nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from graphs.layers.lifting_dwt_nets import DWTPytorchWaveletsLayer,\
    BasicWavelet, AttentionWavelet, AttentionWaveletPostProcessing, LiftingBasedNeuralWaveletv4
from graphs.layers.masked_conv2d import MaskedConv2d
import torch.nn.functional as F
from graphs.layers.upscaling import zeroTreeWaveletPreviousLayer
from compressai.ans import BufferedRansEncoder, RansDecoder
import math

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def byte_extractor(list):
    length = len(list)
    return length

def byte_extractor_xo(list):
    length = 0
    for i in range(len(list)):
        if (len(list[i])<5):
            if (i != len(list)-1):
                for j in range(3):
                    length += len(list[i][j])
            else:
                length += len(list[i])
        else:
            length += len(list[i])

    return length
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class LiftingBasedDWTNetWrapper(nn.Module):
    """ Neural-network architecture that implements ... """
    def __init__(self, config):
        """ Set up sizes/lengths for models, inputs etc """
        super(LiftingBasedDWTNetWrapper, self).__init__()
        self.clrch = config.clrch
        if self.clrch == 3:  # rgb img processed together in 1 NN
            self.model = LiftingBasedDWTNet(config)
        elif self.clrch == 1:  # yuv image color channels processed separately in 3 NN
            self.model0 = LiftingBasedDWTNet(config)
            self.model1 = LiftingBasedDWTNet(config)
            self.model2 = LiftingBasedDWTNet(config)

    def forward(self, x):
        if self.clrch == 3:
            return self.model.forward(x)
        elif self.clrch == 1:
            xhat_0, self_informations_xe_0, self_informations_xo_list_0 = self.model0(x[:, 0:1, :, :])
            xhat_1, self_informations_xe_1, self_informations_xo_list_1 = self.model1(x[:, 1:2, :, :])
            xhat_2, self_informations_xe_2, self_informations_xo_list_2 = self.model2(x[:, 2:3, :, :])
            # combine results into single tensor or list
            xhat = torch.cat((xhat_0, xhat_1, xhat_2), dim=1)
            self_informations_xe = torch.cat((self_informations_xe_0, self_informations_xe_1, self_informations_xe_2), dim=1)
            self_informations_xo_list = []
            self_informations_xo_list.extend(self_informations_xo_list_0)
            self_informations_xo_list.extend(self_informations_xo_list_1)
            self_informations_xo_list.extend(self_informations_xo_list_2)
            return xhat, self_informations_xe, self_informations_xo_list

    def display(self, x):
        pass

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck module(s).
        """
        if self.clrch == 3:
            aux_loss = self.model.aux_loss()
        elif self.clrch == 1:
            aux_loss = self.model0.aux_loss() + self.model1.aux_loss() + self.model2.aux_loss()
        return aux_loss

    def compress(self,x):
        if self.clrch == 3:
            return self.model.compress(x)
        elif self.clrch == 1:
            xhat_0, self_informations_xe_0, self_informations_xo_list_0 = self.model0.compress(x[:, 0:1, :, :])
            xhat_1, self_informations_xe_1, self_informations_xo_list_1 = self.model1.compress(x[:, 1:2, :, :])
            xhat_2, self_informations_xe_2, self_informations_xo_list_2 = self.model2.compress(x[:, 2:3, :, :])
            # combine results into single tensor or list
            xhat = torch.cat((xhat_0, xhat_1, xhat_2), dim=1)
            len_xe_y = byte_extractor(self_informations_xe_0)
            len_xe_cb = byte_extractor(self_informations_xe_1)
            len_xe_cr = byte_extractor(self_informations_xe_2)

            len_xo_y = byte_extractor_xo(self_informations_xo_list_0)
            len_xo_cb = byte_extractor_xo(self_informations_xo_list_1)
            len_xo_cr = byte_extractor_xo(self_informations_xo_list_2)

            len_xe = len_xe_y + len_xe_cb + len_xe_cr
            len_xo = len_xo_y + len_xo_cb + len_xo_cr

            B,C,H,W = x.shape
            len_xe = len_xe*8/(H*W)
            len_xo = len_xo*8/(H*W)
            return xhat, len_xe, len_xo
class LiftingBasedDWTNet(nn.Module):
    """ Neural-network architecture that implements ... """
    def __init__(self, config):
        """ Set up sizes/lengths for models, inputs etc """
        super(LiftingBasedDWTNet, self).__init__()
        self.clrch = config.clrch
        # Transform models
        #classic wavelet
        if config.netType == "CDF97":
            self.autoencoder = DWTPytorchWaveletsLayer(config)
        elif config.netType == "LiftingBasedNeuralWaveletv4":
        # convolutional lifting model with P and U filters
            self.autoencoder = LiftingBasedNeuralWaveletv4(config)
        # convolutional autoencoder type wavelet
        elif config.netType == "BasicWavelet":
            self.autoencoder = BasicWavelet(config)
        # wavelet transform modeled by attention blocks
        elif config.netType == "AttentionWavelet":
            self.autoencoder = AttentionWavelet(config)
        elif config.netType == "AttentionWaveletPostProcessing":
            self.autoencoder = AttentionWaveletPostProcessing(config)

        self.entropy_layer = config.entropy_layer
        #factorized entropy model
        if self.entropy_layer == "factorized":
            self.entropymodel = DWTFactorizedEntropyLayer(config)
        # EZWT levelwise dependency model
        elif self.entropy_layer == "onlyEZWT":
            self.entropymodel = onlyEZWT(config)
        # both causal and EZWT dependency model
        elif self.entropy_layer == "conditioned2ZTsepSubbands":
            self.entropymodel = DWTConditioned2EntropyLayerZTsepSubbands(config)
        # blockwise causal and EZWT dependency model
        elif self.entropy_layer == "DWTConditioned2EntropyLayerZTBlock":
            self.entropymodel = DWTConditioned2EntropyLayerZTBlock(config)

    def compress(self, x):
        """
        See notes if can not easily understand architecture
        :param x: original images/patches  # B x C x H x W
        :return: reconstructed image, -log2(quantized latent tensor probability)
        """
        # check the variable true or not
        if self.entropy_layer == "factorized" or self.entropy_layer == "conditioned2ZTsepSubbands" or self.entropy_layer=="DWTConditioned2EntropyLayerZTBlock"\
                or self.entropy_layer == "onlyEZWT":
            # autoenc, entropy, autodec
            out_xe, out_xo_list = self.autoencoder.encode(x)
            self_informations_xe, self_informations_xo_list, xe_qnt, xo_list_qnt = self.entropymodel.test(out_xe,
                                                                                                          out_xo_list)
            xhat = self.autoencoder.decode(xe_qnt, xo_list_qnt)
        else:
            raise ValueError
        return xhat, self_informations_xe, self_informations_xo_list

    def forward(self, x):
        """
        See notes if can not easily understand architecture
        :param x: original images/patches  # B x C x H x W
        :return: reconstructed image, -log2(quantized latent tensor probability)
        """
        # check the variable true or not
        if self.entropy_layer == "factorized" or self.entropy_layer == "conditioned2ZTsepSubbands" or self.entropy_layer=="DWTConditioned2EntropyLayerZTBlock"\
                or self.entropy_layer == "onlyEZWT":
            # autoenc, entropy, autodec
            out_xe, out_xo_list = self.autoencoder.encode(x)
            self_informations_xe, self_informations_xo_list, xe_qnt, xo_list_qnt = self.entropymodel(out_xe, out_xo_list)
            xhat = self.autoencoder.decode(xe_qnt, xo_list_qnt)
        else:
            raise ValueError

        return xhat, self_informations_xe, self_informations_xo_list

    def display(self, x):
        pass

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum( m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck) )
        return aux_loss
# Factorized model, no condition
class DWTFactorizedEntropyLayer(nn.Module):
    """ Factorized entropy model where each channel/subband is assumed to be independent of other channels and
    each channel is assumed to have a single distribution and each variable in each channel is assumed independent
    """
    def __init__(self, config):
        """
        Estimates entropy during training; performs quantization/rounding during validation.
        """
        super(DWTFactorizedEntropyLayer, self).__init__()
        # number of lifting layers in DWT decomposition
        self.num_lifting_layers = config.dwtlevels
        assert(self.num_lifting_layers > 0)  # at least 1 layer
        # color channels in input; split modes and se so information in each level
        self.clrch = config.clrch
        self.se, self.so = 1,3
        se, so = self.se * self.clrch, self.so * self.clrch
        # define a factorized entropy model for each decomposition layer due to different spatial dimensions
        # define scaling factors for each subband due to integer quantization in entropy layer (one scale per channel)
        self.ent_out_xo_list = nn.ModuleList()
        self.scl_out_xo_list = nn.ParameterList()
        self.scb_out_xo_list = nn.ParameterList()
        for i in range(0, self.num_lifting_layers, 1):
            self.ent_out_xo_list.append(EntropyBottleneck(channels=so))  # 3 subbands
            self.scl_out_xo_list.append(nn.Parameter( nn.init.constant_(torch.empty(1, so, 1, 1), i+1.0) ))  # scaling
            self.scb_out_xo_list.append(nn.Parameter( nn.init.constant_(torch.empty(1, so, 1, 1), 1.0) ))  # backw scaling
            so = se * self.so  # computation or der of this line must be before next line (bec o.w. se changes)
            se = se * self.se
        self.ent_out_xe = EntropyBottleneck(channels=se/self.se)  # 1 subband
        self.scl_out_xe = nn.Parameter(nn.init.constant_(torch.empty(1, int(se/self.se), 1, 1), 5.0))  # scaling params
        self.scb_out_xe = nn.Parameter(nn.init.constant_(torch.empty(1, int(se/self.se), 1, 1), 1.0/5.0))  # backw scaling
        # !!! define scaling params here, apply them in forward at beginning and their inverses at end
        # !!! one scale per channel in each subband

    def forward(self, out_xe, out_xo_list):
        """
        See notes if can not easily understand architecture
        :param out_xe:
        :param out_xo_list:
        :return:
        """
        out_xo_list_qnt = []
        self_informations_out_xo_list = []
        for i in range(0, self.num_lifting_layers, 1):
            out_xo_qnt, pmf_values_out_xo = self.ent_out_xo_list[i].forward(out_xo_list[i])  # quantize
            self_informations_out_xo_list.append(-torch.log2(pmf_values_out_xo))  # bitlengths
            out_xo_list_qnt.append(out_xo_qnt)  # output

        out_xe_qnt, pmf_values_out_xe = self.ent_out_xe.forward(out_xe)  # quantize; everything seems to be done inside here
        self_informations_out_xe = -torch.log2(pmf_values_out_xe)   # bitlengths
        return self_informations_out_xe, self_informations_out_xo_list, out_xe_qnt, out_xo_list_qnt
# Causal + ezwt model
class DWTConditioned2EntropyLayerZTsepSubbands(nn.Module):
    """
    Like DWTConditioned2EntropyLayerZT, but the conditioning is not jointly for subbands but individually like in the
    actual ZeroTree algorithm (i.e. LH-->LH, HL-->HL, HH-->HH. in DWTConditioned2EntropyLayerZT : LH,HL,HH-->LH,HL,HH )
    """
    def __init__(self, config):
        """
        Estimates entropy during training; performs quantization/rounding during validation.
        """
        super(DWTConditioned2EntropyLayerZTsepSubbands, self).__init__()
        # number of lifting layers in DWT decomposition
        self.num_lifting_layers = config.dwtlevels
        self.scale_table = get_scale_table()
        self.config = config
        assert(self.num_lifting_layers > 0)  # at least 1 layer
        # color channels in input; split modes and se so information in each level
        self.clrch = config.clrch
        self.se, self.so = 1,3
        se, so = self.se * self.clrch, self.so * self.clrch
        self.ses = []
        self.sos = []
        for i in range(0, self.num_lifting_layers, 1):
            self.sos.append(so)
            self.ses.append(se)
            so = se * self.so  # computation or der of this line must be before next line (bec o.w. se changes)
            se = se * self.se
        # define a cond.Gauss entropy model for each decomposition layer (except last) xo
        # in compreassai entropy layer (one scale per channel)
        self.plc_list = nn.ModuleList()  # previous_layer_context generating NN
        self.csc_list = nn.ModuleList()   # causal_spatial_context generating NN
        self.cgp_out_xo_list = nn.ModuleList()
        self.ent_out_xo_list = nn.ModuleList()
        # these are not used anymore
        self.scl_out_xo_list = nn.ParameterList()
        self.scb_out_xo_list = nn.ParameterList()
        for i in range(0, self.num_lifting_layers - 1, 1):  # final xo layer coded without conditioning like xe layer
            inn_ch1 = self.sos[i + 1]  # all xo channels in one previous layer
            out_ch1 = inn_ch1 * 81
            self.plc_list.append(nn.Sequential(nn.Conv2d(inn_ch1, out_ch1, kernel_size=3, stride=1, padding=3//2), nn.LeakyReLU(),
                                 nn.Conv2d(out_ch1, out_ch1, kernel_size=3, stride=1, padding=3//2)))
            inn_ch2 = self.sos[i]   # 5x5 causal spatial neighbs
            out_ch2 = inn_ch2 * 81
            self.csc_list.append(
                MaskedConv2d(mask_type='A', in_channels=inn_ch2, out_channels=out_ch2, kernel_size=5, stride=1, padding=5//2, groups=inn_ch2)  # groups=self.sos[i])
            )
            inn_ch = out_ch1 + out_ch2
            out_ch = self.sos[i] * 2  # mu, sigma
            self.cgp_out_xo_list.append(
                nn.Sequential(   # cgp : generate con.gaus. pars mu,sigma for each xo chan from all prev layers' xo's + spat. causal context
                    nn.Conv2d(1 * inn_ch // 1, 1 * inn_ch // 1, kernel_size=1, stride=1, padding=1 // 2,
                              groups=inn_ch1), nn.LeakyReLU(inplace=True),
                    nn.Conv2d(1 * inn_ch // 1, 1 * inn_ch // 3, kernel_size=1, stride=1, padding=1 // 2,
                              groups=inn_ch1), nn.LeakyReLU(inplace=True),
                    nn.Conv2d(1 * inn_ch // 3, 1 * inn_ch // 9, kernel_size=1, stride=1, padding=1 // 2,
                              groups=inn_ch1), nn.LeakyReLU(inplace=True),
                    nn.Conv2d(1 * inn_ch // 9, out_ch, kernel_size=1, stride=1, padding=1 // 2, groups=inn_ch1)
                )
            )
            self.ent_out_xo_list.append(GaussianConditional(scale_table=None, scale_bound=0.11))  # scale+bound default ?

        # now define factorized entropy model for last layer xo and xe
        i = self.num_lifting_layers - 1
        inn_ch2 = self.sos[i]   # 3x3 causal spatial neighbs
        out_ch2 = inn_ch2 * 81
        out_ch = self.sos[i] * 2  # mu, sigma
        self.csc_list.append(
            nn.Sequential(
                MaskedConv2d(mask_type='A', in_channels=inn_ch2,    out_channels=out_ch2,    kernel_size=3, stride=1, padding=3//2, groups=inn_ch2), nn.LeakyReLU(inplace=True),  # groups=self.sos[i])
                MaskedConv2d(mask_type='B', in_channels=out_ch2//1, out_channels=out_ch2//1, kernel_size=3, stride=1, padding=3//2, groups=inn_ch2), nn.LeakyReLU(inplace=True),
                MaskedConv2d(mask_type='B', in_channels=out_ch2//1, out_channels=out_ch2//3, kernel_size=3, stride=1, padding=3//2, groups=inn_ch2), nn.LeakyReLU(inplace=True),
                MaskedConv2d(mask_type='B', in_channels=out_ch2//3, out_channels=out_ch2//9, kernel_size=3, stride=1, padding=3//2, groups=inn_ch2), nn.LeakyReLU(inplace=True),
                MaskedConv2d(mask_type='B', in_channels=out_ch2//9, out_channels=out_ch,     kernel_size=3, stride=1, padding=3//2, groups=inn_ch2)
            )
        )
        self.ent_out_xo_list.append(GaussianConditional(scale_table=None, scale_bound=0.11))
        inn_ch2 = self.ses[i]   # 3x3 causal spatial neighbs
        out_ch2 = inn_ch2 * 81
        out_ch = self.ses[i] * 2  # mu, sigma
        self.csc_xe = nn.Sequential(
            MaskedConv2d(mask_type='A', in_channels=inn_ch2,    out_channels=out_ch2,    kernel_size=3, stride=1, padding=3//2, groups=inn_ch2), nn.LeakyReLU(inplace=True),  # groups=self.sos[i])
            MaskedConv2d(mask_type='B', in_channels=out_ch2//1, out_channels=out_ch2//1, kernel_size=3, stride=1, padding=3//2, groups=inn_ch2), nn.LeakyReLU(inplace=True),
            MaskedConv2d(mask_type='B', in_channels=out_ch2//1, out_channels=out_ch2//3, kernel_size=3, stride=1, padding=3//2, groups=inn_ch2), nn.LeakyReLU(inplace=True),
            MaskedConv2d(mask_type='B', in_channels=out_ch2//3, out_channels=out_ch2//9, kernel_size=3, stride=1, padding=3//2, groups=inn_ch2), nn.LeakyReLU(inplace=True),
            MaskedConv2d(mask_type='B', in_channels=out_ch2//9, out_channels=out_ch,     kernel_size=3, stride=1, padding=3//2, groups=inn_ch2)
        )
        self.ent_out_xe = GaussianConditional(scale_table=None, scale_bound=0.11)  #



    def forward(self, out_xe, out_xo_list):
        """
        :param out_xe:
        :param out_xo_list:
        :param self_ae: autoencoder class
        :return:
        """
        # first, quantize xe channels, get xe entropy
        out_xe_qnt = self.ent_out_xe.quantize(out_xe, "noise" if self.training else "dequantize")
        out_xe_qnt_mu_sigma = self.csc_xe.forward(out_xe_qnt)
        sigma = out_xe_qnt_mu_sigma[:, 0::2, :, :]
        mu    = out_xe_qnt_mu_sigma[:, 1::2, :, :]
        _, pmf_values_out_xe = self.ent_out_xe.forward(out_xe,  sigma, means=mu, training=None)  # quantize, everything seems to be done inside here
        self_informations_out_xe = -torch.log2(pmf_values_out_xe)  # bitlengths
        # last layer xo; quantize, get entropy
        out_xo_list_qnt = []
        self_informations_out_xo_list = []
        i = self.num_lifting_layers - 1
        # get 3x3 causal spatial context information
        out_xo_qnt = self.ent_out_xo_list[i].quantize(out_xo_list[i], "noise" if self.training else "dequantize")
        out_xo_qnt_mu_sigma = self.csc_list[i].forward(out_xo_qnt)
        sigma = out_xo_qnt_mu_sigma[:, 0::2, :, :]
        mu    = out_xo_qnt_mu_sigma[:, 1::2, :, :]
        _, pmf_values_out_xo = self.ent_out_xo_list[i].forward(out_xo_list[i], sigma, means=mu, training=None)
        self_informations_out_xo_list.append(-torch.log2(pmf_values_out_xo))  # bitlengths
        out_xo_list_qnt.append(out_xo_qnt)  # output
        con_channels = out_xo_qnt.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # form channels on which to condition for the xo channels' coding
        # next, xo channels from smallest resolution to highest resolution
        for i in range(self.num_lifting_layers - 2, -1, -1):
            # get causal spatial context information
            out_xo_qnt = self.ent_out_xo_list[i].quantize(out_xo_list[i], "noise" if self.training else "dequantize")
            csc_params = self.csc_list[i].forward(out_xo_qnt)
            # get prev layer context information
            plc_params = self.plc_list[i].forward(con_channels)
            # get con.gauss params sigma,mu
            plc_0, plc_1, plc_2 = plc_params.chunk(3, dim=1)
            csc_0, csc_1, csc_2 = csc_params.chunk(3, dim=1)
            plc_csc_grpd_params = torch.cat((plc_0, csc_0, plc_1, csc_1, plc_2, csc_2), dim=1)
            out_xo_qnt_mu_sigma = self.cgp_out_xo_list[i].forward(plc_csc_grpd_params)
            sigma = out_xo_qnt_mu_sigma[:, 0::2, :, :]
            mu    = out_xo_qnt_mu_sigma[:, 1::2, :, :]
            # quantize xo channel, get xo entropy
            _, pmf_values_out_xo = self.ent_out_xo_list[i].forward(out_xo_list[i], sigma, means=mu, training=None)  # out_xo_qnt, pmf_values_out_xo = self.ent_out_xo_list[i].forward(out_xo_list[i], sigma, means=mu)  # quantize, everything seems to be done inside here
            self_informations_out_xo_list.append(-torch.log2(pmf_values_out_xo))  # bitlengths
            out_xo_list_qnt.append(out_xo_qnt)  # output
            con_channels = out_xo_qnt.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # update xo channels on which to condition
        # last operations before retruning
        out_xo_list_qnt.reverse()
        self_informations_out_xo_list.reverse()
        # return all self_infos and qnt_tensors
        return self_informations_out_xe, self_informations_out_xo_list, out_xe_qnt, out_xo_list_qnt

    def test(self, out_xe, out_xo_list):
        """
        :param out_xe:
        :param out_xo_list:
        :param self_ae: autoencoder class
        :return:
        """
        torch.backends.cudnn.benchmark = True
        xo_dequant_list = []
        kernel_size = 3
        padding_mask = kernel_size//2
        # first, quantize xe channels, get xe entropy
        # out_xe = out_xe * self.scl_out_xe.expand_as(out_xe)  # forward scale
        y = out_xe
        y_hat = F.pad(y, (padding_mask,padding_mask,padding_mask,padding_mask))
        shape = y.shape
        model = self.csc_xe
        self_informations_out_xe, out_xe_qnt = self.compress_ar(y_hat, shape, kernel_size, self.ent_out_xe, model, None, flag="low")
        # last layer xo; quantize, get entropy
        out_xo_list_qnt = []
        self_informations_out_xo_list = []
        y = out_xo_list[self.config.dwtlevels-1]
        shape = y.shape
        y_hat = F.pad(y, (padding_mask, padding_mask, padding_mask, padding_mask))
        model = self.csc_list[(self.config.dwtlevels-1)]
        xo_info,  out_xo_qnt = self.compress_ar(y_hat, shape, kernel_size, self.ent_out_xo_list[self.config.dwtlevels-1], model, None, flag= "high")
        con_channels = out_xo_qnt.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        self_informations_out_xo_list.append(xo_info)
        out_xo_list_qnt.append(out_xo_qnt)
        kernel_size = 5
        padding_mask = kernel_size//2
        for i in range(self.num_lifting_layers - 2, -1, -1):
            model = self.csc_list[i]
            plc_params = self.plc_list[i].forward(con_channels)
            fuse_model = self.cgp_out_xo_list[i]
            ent_model = self.ent_out_xo_list[i]
            y = out_xo_list[i]
            shape = y.shape
            y_hat = F.pad(y, (padding_mask, padding_mask, padding_mask, padding_mask))
            param_hat = F.pad(plc_params, (padding_mask, padding_mask, padding_mask, padding_mask))
            self_information_xo, out_xo_qnt = self.compress_ar(y_hat, shape, kernel_size, ent_model, model, fuse_model, param_hat, flag="high")
            self_informations_out_xo_list.append(self_information_xo)
            con_channels = out_xo_qnt.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
            out_xo_list_qnt.append(out_xo_qnt)
        # last operations before retruning
        out_xo_list_qnt.reverse()
        self_informations_out_xo_list.reverse()
        # return all self_infos and qnt_tensors

        # decode
        kernel_size = 3
        shape = out_xe.shape
        model = self.csc_xe
        out_xe_dequant = self.decompress_ar(self_informations_out_xe, shape, kernel_size, self.ent_out_xe, model, flag="low")

        y = out_xo_list_qnt[self.config.dwtlevels-1]
        shape = y.shape
        self_information_xo = self_informations_out_xo_list[self.config.dwtlevels-1]
        model = self.csc_list[(self.config.dwtlevels-1)]
        out_xo_dequant = self.decompress_ar(self_information_xo, shape, kernel_size, self.ent_out_xo_list[self.config.dwtlevels-1], model, flag="high")
        if self.config.mode != "test":
            con_channels = out_xo_dequant.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).cuda()
        else:
            con_channels = out_xo_dequant.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        xo_dequant_list.append(out_xo_dequant)
        kernel_size = 5
        padding_mask = kernel_size//2
        for i in range(self.num_lifting_layers - 2, -1, -1):
            model = self.csc_list[i]
            plc_params = self.plc_list[i].forward(con_channels)
            fuse_model = self.cgp_out_xo_list[i]
            ent_model = self.ent_out_xo_list[i]
            y = out_xo_list_qnt[i]
            shape = y.shape
            # y_hat = F.pad(y, (padding_mask, padding_mask, padding_mask, padding_mask))
            self_information_xo = self_informations_out_xo_list[i]
            param_hat = F.pad(plc_params, (padding_mask, padding_mask, padding_mask, padding_mask))
            out_xo_deqnt = self.decompress_ar(self_information_xo, shape, kernel_size, ent_model, model, fuse_model, param_hat, flag="high")
            xo_dequant_list.append(out_xo_deqnt)
            con_channels = out_xo_deqnt.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        xo_dequant_list.reverse()

        return self_informations_out_xe, self_informations_out_xo_list, out_xe_dequant, xo_dequant_list

    def compress_ar(self, y_hat, shape, network_kernel_size, emodel, csc, cgp, param=None, flag=None):
        padding = network_kernel_size//2
        B,C,height, width = shape[0], shape[1], shape[2], shape[3]
        # self.ent_out_xe.update_scale_table(self.scale_table)
        emodel.update_scale_table(scale_table=self.scale_table)
        cdf = emodel.quantized_cdf.tolist()
        cdf_lengths = emodel.cdf_length.tolist()
        offsets = emodel.offset.tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + network_kernel_size, w: w + network_kernel_size]
                csc_params = csc(y_crop)
                if (param != None):
                    plc_0, plc_1, plc_2 = param[:, :, h:h+network_kernel_size, w:w+network_kernel_size].chunk(3, dim=1)
                    csc_0, csc_1, csc_2 = csc_params.chunk(3, dim=1)
                    plc_csc_grpd_params = torch.cat((plc_0, csc_0, plc_1, csc_1, plc_2, csc_2), dim=1)
                    out_xo_qnt_mu_sigma = cgp.forward(plc_csc_grpd_params)
                    sigma = out_xo_qnt_mu_sigma[:, 0::2, :, :]
                    mu = out_xo_qnt_mu_sigma[:, 1::2, :, :]
                else:
                    sigma = csc_params[:, 0::2, :, :]
                    mu = csc_params[:, 1::2, :, :]
                sigma_val = sigma[:,:,padding, padding]
                mu_val = mu[:,:,padding, padding]
                indexes = emodel.build_indexes(sigma_val)
                y_crop = y_crop[:, :, padding, padding]
                y_q = emodel.quantize(y_crop, "symbols", mu_val)
                y_hat[:, :, h + padding, w + padding] = y_q + mu_val
                if (flag == "low"):
                    yq_list = []
                    yq_list.append(y_q)
                    index_list = []
                    index_list.append(indexes)
                    symbols_list.extend(yq_list)
                    indexes_list.extend(index_list)
                elif (flag == "high"):
                    symbols_list.extend(y_q.squeeze().tolist())
                    indexes_list.extend(indexes.squeeze().tolist())


        print("done_comp")
        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_hat = F.pad(y_hat, (-padding, - padding, -padding, -padding))
        string = encoder.flush()
        return string, y_hat

    def decompress_ar(self, strings, shape, kernel_size, emodel, csc= None, cgp=None, param=None, flag = None):
        kernel_size = kernel_size
        padding = kernel_size//2
        y_hat = torch.zeros((shape[0],shape[1], shape[2]+ 2 * padding, shape[3]+ 2 * padding))
        entropy_model = emodel
        cdf = entropy_model.quantized_cdf.tolist()
        cdf_lengths = entropy_model.cdf_length.tolist()
        offsets = entropy_model.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings)
        height = shape[2]
        width = shape[3]
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                csc_params = csc(y_crop)
                if (param != None):
                    plc_0, plc_1, plc_2 = param[:, :, h:h+kernel_size, w:w+kernel_size].chunk(3, dim=1)
                    csc_0, csc_1, csc_2 = csc_params.chunk(3, dim=1)
                    plc_csc_grpd_params = torch.cat((plc_0, csc_0, plc_1, csc_1, plc_2, csc_2), dim=1)
                    out_xo_qnt_mu_sigma = cgp.forward(plc_csc_grpd_params)
                    sigma = out_xo_qnt_mu_sigma[:, 0::2, :, :]
                    mu = out_xo_qnt_mu_sigma[:, 1::2, :, :]
                else:
                    sigma = csc_params[:, 0::2, :, :]
                    mu = csc_params[:, 1::2, :, :]
                sigma_val = sigma[:, :, padding:padding+1, padding:padding+1]
                mu_val = mu[:, :, padding:padding+1, padding:padding+1]
                indexes = entropy_model.build_indexes(sigma_val)
                if (flag=="low"):
                    list_form = []
                    list_form.append(indexes.squeeze())
                    rv = decoder.decode_stream(
                        list_form, cdf, cdf_lengths, offsets
                    )
                elif (flag =="high"):
                    rv = decoder.decode_stream(
                        indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                    )

                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = entropy_model.dequantize(rv, mu_val)
                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp:hp+1, wp:wp+1] = rv
        decompressed = F.pad(y_hat,(-padding, -padding, - padding, -padding))
        # decompressed = y_hat
        print("done_decomp")
        return decompressed
# Block based ezwt approach
class DWTConditioned2EntropyLayerZTBlock(nn.Module):
    """
    Like DWTConditioned2EntropyLayerZT, but the conditioning is not jointly for subbands but individually like in the
    actual ZeroTree algorithm (i.e. LH-->LH, HL-->HL, HH-->HH. in DWTConditioned2EntropyLayerZT : LH,HL,HH-->LH,HL,HH )
    """

    def __init__(self, config):
        """
        Estimates entropy during training; performs quantization/rounding during validation.
        """
        super(DWTConditioned2EntropyLayerZTBlock, self).__init__()
        # number of lifting layers in DWT decomposition
        self.num_lifting_layers = config.dwtlevels
        self.dwtLevels = config.dwtlevels
        assert (self.num_lifting_layers > 0)  # at least 1 layer
        # color channels in input; split modes and se so information in each level
        self.clrch = config.clrch
        self.multiplier = 8
        self.se = 1
        self.so = 3
        se, so = self.se * self.clrch, self.so * self.clrch
        self.ses = []
        self.sos = []
        for i in range(0, self.num_lifting_layers, 1):
            self.sos.append(so)
            self.ses.append(se)
            so = se * self.so  # computation or der of this line must be before next line (bec o.w. se changes)
            se = se * self.se

        # define a cond.Gauss entropy model for each decomposition layer (except last) xo
        # define scaling factors for each subband due to integer quantization
        # in compreassai entropy layer (one scale per channel)
        self.dep_1_list_mu = nn.ModuleList()
        self.dep_2_list_mu = nn.ModuleList()
        self.dep_3_list_mu = nn.ModuleList()
        self.dep_4_list_mu = nn.ModuleList()

        self.dep_1_list_sigma = nn.ModuleList()  # previous_layer_context generating NN
        self.dep_2_list_sigma = nn.ModuleList()
        self.dep_3_list_sigma = nn.ModuleList()
        self.dep_4_list_sigma = nn.ModuleList()

        self.cgp_out_xo_list = nn.ModuleList()
        self.ent_out_xo_list = nn.ModuleList()
        self.scl_out_xo_list = nn.ParameterList()
        self.scb_out_xo_list = nn.ParameterList()
        for i in range(0, self.num_lifting_layers - 1, 1):  # final xo layer coded without conditioing like xe layer
            for j in range(3):
                self.ent_out_xo_list.append(
                    GaussianConditional(scale_table=None, scale_bound=0.11))  # scale+bound default ?
                self.scl_out_xo_list.append(nn.Parameter(
                    nn.init.constant_(torch.empty(1, self.sos[i], 1, 1), i * 1.0 + 1.0)))  # scaling per channel
                self.scb_out_xo_list.append(nn.Parameter(
                    nn.init.constant_(torch.empty(1, self.sos[i], 1, 1), 1.0)))  # backw scaling per channel
                in_ch_1 = 1
                in_ch_2 = 2
                in_ch_3 = 3
                in_ch_4 = 4
                hid_ch = 32

                self.dep_1_list_mu.append(nn.Sequential(
                        nn.Conv2d( in_ch_1,   hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, 1,          kernel_size=1, stride=1, padding=1 // 2)
                    ))

                self.dep_2_list_mu.append(nn.Sequential(
                        nn.Conv2d( in_ch_2,   hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, 1,          kernel_size=1, stride=1, padding=1 // 2)
                    ))

                self.dep_3_list_mu.append(nn.Sequential(
                        nn.Conv2d( in_ch_3,   hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, 1,          kernel_size=1, stride=1, padding=1 // 2)
                    ))

                self.dep_4_list_mu.append(nn.Sequential(
                        nn.Conv2d( in_ch_4,   hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, 1,          kernel_size=1, stride=1, padding=1 // 2)
                    ))

                self.dep_1_list_sigma.append(nn.Sequential(
                        nn.Conv2d( in_ch_1,   hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, 1,          kernel_size=1, stride=1, padding=1 // 2)
                    ))

                self.dep_2_list_sigma.append(nn.Sequential(
                        nn.Conv2d( in_ch_2,   hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, 1,          kernel_size=1, stride=1, padding=1 // 2)
                    ))

                self.dep_3_list_sigma.append(nn.Sequential(
                        nn.Conv2d( in_ch_3,   hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, 1,          kernel_size=1, stride=1, padding=1 // 2)
                    ))

                self.dep_4_list_sigma.append(nn.Sequential(
                        nn.Conv2d( in_ch_4,   hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=3, stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, hid_ch, kernel_size=1, stride=1, padding=1 // 2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d( hid_ch, 1,          kernel_size=1, stride=1, padding=1 // 2)
                    ))





        self.ent_out_xo_list.append(GaussianConditional(scale_table=None, scale_bound=0.11))
        # self.ent_out_xe = GaussianConditional(scale_table=None, scale_bound=0.11)  #
        self.gaussian_conditional = GaussianConditional(None)
        self.ent_out_xe = EntropyBottleneck(channels=1)
        self.ent_out_xo = EntropyBottleneck(channels=3)
    def forward(self, out_xe, out_xo_list):
        """
        :param out_xe:
        :param out_xo_list:
        :param self_ae: autoencoder class
        :return:
        """
        out_xo_list_qnt = []
        self_informations_out_xo_list = []
        # first, quantize xe channels, get xe entropy
        out_xe_qnt, pmf_values_out_xe = self.ent_out_xe.forward(out_xe)  # quantize; everything seems to be done inside here
        self_informations_out_xe = -torch.log2(pmf_values_out_xe)
        # quantize, everything seems to be done inside here
        i = self.num_lifting_layers - 1
        out_xo_qnt, pmf_values_out_xo = self.ent_out_xo.forward(out_xo_list[i])  # quantize; everything seems to be done inside here
        self_informations_out_xo = -torch.log2(pmf_values_out_xo)
        self_informations_out_xo_list.append(self_informations_out_xo)
        out_xo_list_qnt.append(out_xo_qnt)
        con_channels = out_xo_qnt
        # next, xo channels from smallest resolution to highest resolution
        for i in range(0, self.num_lifting_layers - 1, 1):
            self_informations_out_xo_list_dummy = []
            out_xo_list_qnt_dummy = []
            for j in range(3):
                # get causal spatial context information
                B, C, H, W = out_xo_list[self.dwtLevels - i - 2].shape
                mu_array = torch.empty((B,1,H,W)).cuda()
                sigma_array = torch.empty((B,1,H,W)).cuda()
                out_xo_qnt = self.ent_out_xo_list[(self.dwtLevels - i - 1) * 3 - j - 1].quantize(
                    out_xo_list[self.dwtLevels - i - 2][:, j:j+1, :, :],
                    "noise" if self.training else "dequantize")
                img_ee = out_xo_qnt[:,:,0::2, 0::2]
                img_eo = out_xo_qnt[:,:,0::2, 1::2]
                img_oe = out_xo_qnt[:,:,1::2, 0::2]

                dep_1 = con_channels[:,j:j+1,:,:]
                mu_array[:,:,0::2,0::2] = self.dep_1_list_mu[j + i*3](dep_1)
                sigma_array[:,:,0::2,0::2] = self.dep_1_list_sigma[j + i*3](dep_1)

                dep_2 = torch.cat((dep_1, img_ee), dim=1)
                mu_array[:,:,0::2,1::2] = self.dep_2_list_mu[j + i*3](dep_2)
                sigma_array[:,:,0::2,1::2] = self.dep_2_list_sigma[j + i*3](dep_2)

                dep_3 = torch.cat((dep_1, img_ee, img_eo), dim=1)
                mu_array[:,:,1::2,0::2] = self.dep_3_list_mu[j + i*3](dep_3)
                sigma_array[:,:,1::2,0::2] = self.dep_3_list_sigma[j + i*3](dep_3)

                dep_4 = torch.cat((dep_1, img_ee, img_eo, img_oe), dim=1)
                mu_array[:,:,1::2,1::2] = self.dep_4_list_mu[j + i*3](dep_4)
                sigma_array[:,:,1::2,1::2] = self.dep_4_list_sigma[j + i*3](dep_4)

                # quantize xo channel, get xo entropy
                _, pmf_values_out_xo_high = self.ent_out_xo_list[(self.dwtLevels - i - 1) * 3 - j - 1].forward(
                    out_xo_list[self.dwtLevels - i - 2][:, j:j+1, :, :], sigma_array, means=mu_array, training=None)
                self_informations_out_xo_list_dummy.append(-torch.log2(pmf_values_out_xo_high))
                out_xo_list_qnt_dummy.append(out_xo_qnt)

            self_informations_out_xo_list.append(torch.cat((self_informations_out_xo_list_dummy[0],
                                                            self_informations_out_xo_list_dummy[1],
                                                            self_informations_out_xo_list_dummy[2]), dim=1))
            con_channels = torch.cat((out_xo_list_qnt_dummy[0], out_xo_list_qnt_dummy[1], out_xo_list_qnt_dummy[2]),
                                     dim=1)
            out_xo_list_qnt.append(con_channels)
        out_xo_list_qnt.reverse()
        self_informations_out_xo_list.reverse()
        # return all self_infos and qnt_tensors
        return self_informations_out_xe, self_informations_out_xo_list, out_xe_qnt, out_xo_list_qnt
# last level factorized, other levels ezwt
class onlyEZWT(nn.Module):
    """

    """
    def __init__(self, config):
        """
        Estimates entropy during training; performs quantization/rounding during validation.
        """
        super(onlyEZWT, self).__init__()
        # number of lifting layers in DWT decomposition
        self.num_lifting_layers = config.dwtlevels
        self.scale_table = get_scale_table()
        self.config = config
        assert(self.num_lifting_layers > 0)  # at least 1 layer
        # color channels in input; split modes and se so information in each level
        self.clrch = config.clrch
        self.se, self.so = 1,3
        se, so = self.se * self.clrch, self.so * self.clrch
        self.ses = []
        self.sos = []
        for i in range(0, self.num_lifting_layers, 1):
            self.sos.append(so)
            self.ses.append(se)
            so = se * self.so  # computation or der of this line must be before next line (bec o.w. se changes)
            se = se * self.se


        self.plc_list = nn.ModuleList()  # previous_layer_context generating NN
        self.ent_out_xo_list = nn.ModuleList()
        for i in range(0, self.num_lifting_layers - 1, 1):  # final xo layer coded without conditioing like xe layer
            # inn_ch1 = sum(self.sos[i+1:self.num_lifting_layers])   # all xo channels in all previous layers
            inn_ch1 = self.sos[i + 1]  # all xo channels in one previous layer
            out_ch1 = inn_ch1 * 81
            # self.plc_list.append(nn.Conv2d(inn_ch1, out_ch1, kernel_size=1, stride=1, padding=1//2, groups=inn_ch1))
            self.plc_list.append(nn.Sequential(nn.Conv2d(inn_ch1, out_ch1, kernel_size=3, stride=1, padding=3//2), nn.LeakyReLU(),
                                 nn.Conv2d(out_ch1, out_ch1, kernel_size=3, stride=1, padding=3//2), nn.LeakyReLU(),
                                               nn.Conv2d(out_ch1, 6, kernel_size=1, stride=1, padding=1//2)))

            self.ent_out_xo_list.append(GaussianConditional(scale_table=None, scale_bound=0.11))  # scale+bound default ?

        # now define factorized entropy model for last layer xo and xe
        self.ent_out_xe = EntropyBottleneck(channels=1)
        self.ent_out_xo = EntropyBottleneck(channels=3)


    def forward(self, out_xe, out_xo_list):
        """
        :param out_xe:
        :param out_xo_list:
        :param self_ae: autoencoder class
        :return:
        """
        # first, quantize xe channels, get xe entropy
        #out_xe = out_xe * self.scl_out_xe.expand_as(out_xe)  # forward scale
        self_informations_out_xo_list = []
        out_xo_list_qnt = []
        out_xe_qnt, pmf_values_out_xe = self.ent_out_xe.forward(out_xe)  # quantize; everything seems to be done inside here
        self_informations_out_xe = -torch.log2(pmf_values_out_xe)
        i = self.num_lifting_layers - 1
        out_xo_qnt, pmf_values_out_xo = self.ent_out_xo.forward(out_xo_list[i])  # quantize; everything seems to be done inside here
        self_informations_out_xo = -torch.log2(pmf_values_out_xo)
        self_informations_out_xo_list.append(self_informations_out_xo)
        out_xo_list_qnt.append(out_xo_qnt)
        con_channels = out_xo_qnt.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        # next, xo channels from smallest resolution to highest resolution
        for i in range(self.num_lifting_layers - 2, -1, -1):
            # get causal spatial context information
            # get prev layer context information
            plc_params = self.plc_list[i].forward(con_channels)
            # get con.gauss params sigma,mu
            sigma = plc_params[:, 0::2, :, :]
            mu    = plc_params[:, 1::2, :, :]
            # quantize xo channel, get xo entropy
            out_xo_qnt, pmf_values_out_xo = self.ent_out_xo_list[i].forward(out_xo_list[i], sigma, means=mu, training=None)  # out_xo_qnt, pmf_values_out_xo = self.ent_out_xo_list[i].forward(out_xo_list[i], sigma, means=mu)  # quantize, everything seems to be done inside here
            self_informations_out_xo_list.append(-torch.log2(pmf_values_out_xo))  # bitlengths
            out_xo_list_qnt.append(out_xo_qnt)  # output
            con_channels = out_xo_qnt.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # update xo channels on which to condition
        # last operations before retruning
        out_xo_list_qnt.reverse()
        self_informations_out_xo_list.reverse()
        # return all self_infos and qnt_tensors
        return self_informations_out_xe, self_informations_out_xo_list, out_xe_qnt, out_xo_list_qnt
# wrong model
class DWTConditioned2EntropyLayerZTsepSubbandsBerk(nn.Module):
    """
    Like DWTConditioned2EntropyLayerZT, but the conditioning is not jointly for subbands but individually like in the
    actual ZeroTree algorithm (i.e. LH-->LH, HL-->HL, HH-->HH. in DWTConditioned2EntropyLayerZT : LH,HL,HH-->LH,HL,HH )
    """
    def __init__(self, config):
        """
        Estimates entropy during training; performs quantization/rounding during validation.
        """
        super(DWTConditioned2EntropyLayerZTsepSubbandsBerk, self).__init__()
        # number of lifting layers in DWT decomposition
        self.num_lifting_layers = config.dwtlevels
        self.dwtLevels = config.dwtlevels
        assert(self.num_lifting_layers > 0)  # at least 1 layer
        # color channels in input; split modes and se so information in each level
        self.clrch = config.clrch
        self.multiplier = 8
        self.se=1
        self.so=3
        se, so = self.se * self.clrch, self.so * self.clrch
        self.ses = []
        self.sos = []
        for i in range(0, self.num_lifting_layers, 1):
            self.sos.append(so)
            self.ses.append(se)
            so = se * self.so  # computation or der of this line must be before next line (bec o.w. se changes)
            se = se * self.se

        # define a cond.Gauss entropy model for each decomposition layer (except last) xo
        # define scaling factors for each subband due to integer quantization
        # in compreassai entropy layer (one scale per channel)
        self.plc_list = nn.ModuleList()  # previous_layer_context generating NN
        self.csc_list = nn.ModuleList()   # causal_spatial_context generating NN
        self.cgp_out_xo_list = nn.ModuleList()
        self.ent_out_xo_list = nn.ModuleList()
        self.scl_out_xo_list = nn.ParameterList()
        self.scb_out_xo_list = nn.ParameterList()
        for i in range(0, self.num_lifting_layers - 1, 1):  # final xo layer coded without conditioing like xe layer
            # inn_ch1 = sum(self.sos[i+1:self.num_lifting_layers])   # all xo channels in all previous layers
            for j in range(3):
                inn_ch1 = 1
                out_ch1 = inn_ch1 * self.multiplier
                self.plc_list.append(zeroTreeWaveletPreviousLayer(config,out_ch1))
                inn_ch2 = 1  # 5x5 causal spatial neighbs
                out_ch2 = inn_ch2 * self.multiplier
                self.csc_list.append(
                    MaskedConv2d(mask_type='A', in_channels=inn_ch2, out_channels=out_ch2, kernel_size=5, stride=1, padding=5//2) # groups=self.sos[i])
                )
                inn_ch = out_ch1 + out_ch2
                out_ch =  2  # mu, sigma
                self.cgp_out_xo_list.append(
                    nn.Sequential(   # cgp : generate con.gaus. pars mu,sigma for each xo chan from all prev layers' xo's + spat. causal context
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(1 * inn_ch//1, 1 * inn_ch//1, kernel_size=1, stride=1, padding=1//2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d(1 * inn_ch//1, 1 * inn_ch//2, kernel_size=1, stride=1, padding=1//2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d(1 * inn_ch//2, 1 * inn_ch//4, kernel_size=1, stride=1, padding=1//2), nn.LeakyReLU(inplace=True),
                        nn.Conv2d(1 * inn_ch//4, out_ch,        kernel_size=1, stride=1, padding=1//2)
                    )
                )
                self.ent_out_xo_list.append(GaussianConditional(scale_table=None, scale_bound=0.11))  # scale+bound default ?
                self.scl_out_xo_list.append(nn.Parameter(nn.init.constant_(torch.empty(1, self.sos[i], 1, 1), i*1.0+1.0)))  # scaling per channel
                self.scb_out_xo_list.append(nn.Parameter(nn.init.constant_(torch.empty(1, self.sos[i], 1, 1), 1.0)))  # backw scaling per channel
        # now define factorized entropy model for last layer xo and xe
        i = self.num_lifting_layers - 1
        inn_ch2 = 3   # 3x3 causal spatial neighbs
        out_ch2 = inn_ch2 * self.multiplier
        out_ch = self.sos[i] * 2  # mu, sigma
        self.csc_list.append(
            nn.Sequential(
                MaskedConv2d(mask_type='A', in_channels=inn_ch2, out_channels=out_ch2, kernel_size=3, stride=1,
                             padding=3 // 2), nn.LeakyReLU(inplace=True),  # groups=self.sos[i])
                MaskedConv2d(mask_type='B', in_channels=out_ch2 // 1, out_channels=out_ch2 // 2, kernel_size=3,
                             stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                MaskedConv2d(mask_type='B', in_channels=out_ch2 // 2, out_channels=out_ch2 // 4, kernel_size=3,
                             stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                MaskedConv2d(mask_type='B', in_channels=out_ch2 // 4, out_channels=out_ch2 // 4, kernel_size=3,
                             stride=1, padding=3 // 2), nn.LeakyReLU(inplace=True),
                MaskedConv2d(mask_type='B', in_channels=out_ch2 // 4, out_channels=out_ch, kernel_size=3, stride=1,
                             padding=3 // 2), nn.LeakyReLU(inplace=True),
            )
        )
        self.ent_out_xo_list.append(GaussianConditional(scale_table=None, scale_bound=0.11))
        self.scl_out_xo_list.append(nn.Parameter(nn.init.constant_(torch.empty(1, self.sos[i], 1, 1), i * 1.0 + 1.0)))  # scaling per channel
        self.scb_out_xo_list.append(nn.Parameter(nn.init.constant_(torch.empty(1, self.sos[i], 1, 1), 1.0)))  # backw scaling per channel
        inn_ch2 = 1   # 3x3 causal spatial neighbs
        out_ch2 = inn_ch2 * self.multiplier
        out_ch = self.ses[i] * 2  # mu, sigma
        self.csc_xe = nn.Sequential(
            MaskedConv2d(mask_type='A', in_channels=inn_ch2,    out_channels=out_ch2,    kernel_size=3, stride=1, padding=3//2), nn.LeakyReLU(inplace=True),  # groups=self.sos[i])
            MaskedConv2d(mask_type='B', in_channels=out_ch2//1, out_channels=out_ch2//2, kernel_size=3, stride=1, padding=3//2), nn.LeakyReLU(inplace=True),
            MaskedConv2d(mask_type='B', in_channels=out_ch2//2, out_channels=out_ch2//4, kernel_size=3, stride=1, padding=3//2), nn.LeakyReLU(inplace=True),
            MaskedConv2d(mask_type='B', in_channels=out_ch2//4, out_channels=out_ch2//4, kernel_size=3, stride=1, padding=3//2), nn.LeakyReLU(inplace=True),
            MaskedConv2d(mask_type='B', in_channels=out_ch2//4, out_channels=out_ch,     kernel_size=3, stride=1, padding=3//2), nn.LeakyReLU(inplace=True),
        )
        self.ent_out_xe = GaussianConditional(scale_table=None, scale_bound=0.11)  #
        self.scl_out_xe = nn.Parameter(nn.init.constant_(torch.empty(1, self.ses[i], 1, 1), 5.0))  # scaling params
        self.scb_out_xe = nn.Parameter(nn.init.constant_(torch.empty(1, self.ses[i], 1, 1), 1.0))  # backw scaling
        self.gaussian_conditional = GaussianConditional(None)
    def forward(self, out_xe, out_xo_list):
        """
        :param out_xe:
        :param out_xo_list:
        :param self_ae: autoencoder class
        :return:
        """
        # first, quantize xe channels, get xe entropy
        #out_xe = out_xe * self.scl_out_xe.expand_as(out_xe)  # forward scale
        out_xe_qnt = self.ent_out_xe.quantize(out_xe, "noise" if self.training else "dequantize")
        out_xe_qnt_mu_sigma = self.csc_xe.forward(out_xe_qnt)
        B,C,H,W = out_xe.shape
        sigma = out_xe_qnt_mu_sigma[:, 0, :, :].view(B,C,H,W)
        mu    = out_xe_qnt_mu_sigma[:, 1:, :, :]
        # deneme, pmf_values_out_xe = self.ent_out_xe.forward(out_xe,  sigma, means=None, training=None) Berk
        # quantize, everything seems to be done inside here
        deneme, pmf_values_out_xe = self.ent_out_xe.forward(out_xe_qnt,  sigma, means=None, training=None)  # quantize, everything seems to be done inside here
        self_informations_out_xe = -torch.log2(pmf_values_out_xe)  # bitlengths
        #out_xe_qnt = out_xe_qnt * self.scb_out_xe.expand_as(out_xe_qnt)  # backwards scale, output
        # last layer xo; quantize, get entropy
        out_xo_list_qnt = []
        self_informations_out_xo_list = []

        i = self.num_lifting_layers - 1
        #out_xo_list[i] = out_xo_list[i] * self.scl_out_xo_list[i].expand_as(out_xo_list[i])  # forward scale
        # get 3x3 causal spatial context information
        out_xo_qnt = self.ent_out_xo_list[i].quantize(out_xo_list[i], "noise" if self.training else "dequantize")
        out_xo_qnt_mu_sigma = self.csc_list[i*3].forward(out_xo_qnt)
        sigma = out_xo_qnt_mu_sigma[:, 0:3, :, :]
        mu    = out_xo_qnt_mu_sigma[:, 3:, :, :]
        # _, pmf_values_out_xo = self.ent_out_xo_list[i*3].forward(out_xo_list[i], sigma, means=None, training=None) Berk
        _, pmf_values_out_xo = self.ent_out_xo_list[i*3].forward(out_xo_qnt, sigma, means=None, training=None)
        self_informations_out_xo_list.append(-torch.log2(pmf_values_out_xo))  # bitlengths
        #out_xo_qnt = out_xo_qnt * self.scb_out_xo_list[i].expand_as(out_xo_qnt)  # backwards scale
        out_xo_list_qnt.append(out_xo_qnt)  # output
        con_channels = out_xo_qnt
        # con_channels = out_xo_qnt.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # form channels on which to condition for the xo channels' coding
        # next, xo channels from smallest resolution to highest resolution
        for i in range(0,self.num_lifting_layers - 1,  1):
            self_informations_out_xo_list_dummy = []
            out_xo_list_qnt_dummy=[]
            for j in range(3):
            # get causal spatial context information
                B,C,H,W = out_xo_list[self.dwtLevels-i-2].shape
                out_xo_qnt = self.ent_out_xo_list[(self.dwtLevels-i-1)*3-j-1].quantize(out_xo_list[self.dwtLevels-i-2][:,j,:,:].view(B,1,H,W), "noise" if self.training else "dequantize")
                csc_params = self.csc_list[(self.dwtLevels-i-1)*3-j-1].forward(out_xo_qnt)
                # get prev layer context information
                plc_params = self.plc_list[(self.dwtLevels-i-1)*3-j-1].forward(con_channels[:,j,:,:].view(B,1,int(H/2),int(W/2)))
                # get con.gauss params sigma,mu
                plc_csc_grpd_params = torch.cat((plc_params,csc_params),dim=1)
                out_xo_qnt_mu_sigma = self.cgp_out_xo_list[(self.dwtLevels-i-1)*3-j-1].forward(plc_csc_grpd_params)
                sigma = out_xo_qnt_mu_sigma[:, 0, :, :].view(B,1,H,W)
                mu    = out_xo_qnt_mu_sigma[:, 1:, :, :].view(B,1,H,W)
                # quantize xo channel, get xo entropy
                #out_xo_list[i] = out_xo_list[i] * self.scl_out_xo_list[i].expand_as(out_xo_list[i])  # forward scale
                # _, pmf_values_out_xo_high = self.ent_out_xo_list[(self.dwtLevels-i-1)*3-j-1].forward(out_xo_list[self.dwtLevels-i-2][:,j,:,:].view(B,1,H,W), sigma, means=None, training=None) Berk
                _, pmf_values_out_xo_high = self.ent_out_xo_list[(self.dwtLevels - i - 1) * 3 - j - 1].forward(out_xo_qnt, sigma, means=None, training=None)
                self_informations_out_xo_list_dummy.append(-torch.log2(pmf_values_out_xo_high))  # bitlengths
                #out_xo_qnt = out_xo_qnt * self.scb_out_xo_list[i].expand_as(out_xo_qnt)  # backwards scale
                out_xo_list_qnt_dummy.append(out_xo_qnt)
                # con_channels = torch.cat((con_channels, out_xo_qnt), dim=1).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # update xo channels on which to condition

            self_informations_out_xo_list.append(torch.cat((self_informations_out_xo_list_dummy[0],self_informations_out_xo_list_dummy[1],self_informations_out_xo_list_dummy[2]),dim=1))
            con_channels = torch.cat((out_xo_list_qnt_dummy[0],out_xo_list_qnt_dummy[1],out_xo_list_qnt_dummy[2]),dim=1)
            out_xo_list_qnt.append(con_channels)
        # last operations before retruning
        out_xo_list_qnt.reverse()
        self_informations_out_xo_list.reverse()
        # return all self_infos and qnt_tensors
        return self_informations_out_xe, self_informations_out_xo_list, out_xe_qnt, out_xo_list_qnt

    def test(self, out_xe, out_xo_list):
        """
        :param out_xe:
        :param out_xo_list:
        :param self_ae: autoencoder class
        :return:
        """
        xo_indexes_list = []
        xe_indexes_list = []
        byte_len_xe = 0
        byte_len_xo = 0
        padding_filter = 5
        padding_mask = padding_filter//2
        # first, quantize xe channels, get xe entropy
        # out_xe = out_xe * self.scl_out_xe.expand_as(out_xe)  # forward scale
        y = out_xe
        y_hat = F.pad(y, (padding_mask,padding_mask,padding_mask,padding_mask))
        shape = y.shape()
        model = self.csc_xe
        self_informations_out_xe = self.compress_ar(y_hat, shape, 5, model, None)
        byte_len_xe += len(self_informations_out_xe[0])

        # deneme = self.decompress(self_informations_out_xe, out_xe.shape)

        # out_xe_qnt = out_xe_qnt * self.scb_out_xe.expand_as(out_xe_qnt)  # backwards scale, output
        # last layer xo; quantize, get entropy
        out_xo_list_qnt = []
        self_informations_out_xo_list = []
        y = out_xo_list[self.config.dwtlevels-1]
        shape = y.shape()
        y_hat = F.pad(y, (padding_mask, padding_mask, padding_mask, padding_mask))
        model = self.csc_list[(self.config.dwtlevels-1)*3]
        self_informations_out_xo_list.append(self.compress_ar(y_hat, shape, 5, model, None))
        for i in range(self.config.dwtlevels):
            for j in range(3):
                parameters = self.csc


        i = self.num_lifting_layers - 1
        # out_xo_list[i] = out_xo_list[i] * self.scl_out_xo_list[i].expand_as(out_xo_list[i])  # forward scale
        # get 3x3 causal spatial context information
        out_xo_qnt = self.ent_out_xo_list[i].quantize(out_xo_list[i], mode="symbols")
        out_xo_qnt_mu_sigma = self.csc_list[i * 3].forward(out_xo_qnt.float())
        sigma = out_xo_qnt_mu_sigma[:, 0:3, :, :]
        mu = out_xo_qnt_mu_sigma[:, 3:, :, :]
        # _, pmf_values_out_xo = self.ent_out_xo_list[i*3].forward(out_xo_list[i], sigma, means=None, training=None) Berk
        self.ent_out_xo_list[i * 3].update_scale_table(self.scale_table)
        xo_indexes = self.ent_out_xo_list[i * 3].build_indexes(sigma)
        xo_indexes_list.append(xo_indexes)
        symbols_xo = self.ent_out_xo_list[i * 3].compress(out_xo_list[i], xo_indexes)
        byte_len_xo += len(symbols_xo[0])
        # out_xo_qnt = out_xo_qnt * self.scb_out_xo_list[i].expand_as(out_xo_qnt)  # backwards scale
        out_xo_list_qnt.append(out_xo_qnt)  # output
        self_informations_out_xo_list.append(symbols_xo)
        con_channels = out_xo_qnt
        # con_channels = out_xo_qnt.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # form channels on which to condition for the xo channels' coding
        # next, xo channels from smallest resolution to highest resolution
        for i in range(0, self.num_lifting_layers - 1, 1):
            self_informations_out_xo_list_dummy = []
            out_xo_list_qnt_dummy = []
            xo_indexes_dummy = []
            for j in range(3):
                # get causal spatial context information
                B, C, H, W = out_xo_list[self.dwtLevels - i - 2].shape
                out_xo_qnt = self.ent_out_xo_list[(self.dwtLevels - i - 1) * 3 - j - 1].quantize(
                    out_xo_list[self.dwtLevels - i - 2][:, j, :, :].view(B, 1, H, W), mode="symbols")
                csc_params = self.csc_list[(self.dwtLevels - i - 1) * 3 - j - 1].forward(out_xo_qnt.float())
                # get prev layer context information
                plc_params = self.plc_list[(self.dwtLevels - i - 1) * 3 - j - 1].forward(
                    con_channels[:, j, :, :].float().view(B, 1, int(H / 2), int(W / 2)))
                # get con.gauss params sigma,mu
                plc_csc_grpd_params = torch.cat((plc_params, csc_params), dim=1)
                out_xo_qnt_mu_sigma = self.cgp_out_xo_list[(self.dwtLevels - i - 1) * 3 - j - 1].forward(
                    plc_csc_grpd_params)
                sigma = out_xo_qnt_mu_sigma[:, 0, :, :].view(B, 1, H, W)
                mu = out_xo_qnt_mu_sigma[:, 1:, :, :].view(B, 1, H, W)

                # quantize xo channel, get xo entropy
                # out_xo_list[i] = out_xo_list[i] * self.scl_out_xo_list[i].expand_as(out_xo_list[i])  # forward scale
                # _, pmf_values_out_xo_high = self.ent_out_xo_list[(self.dwtLevels-i-1)*3-j-1].forward(out_xo_list[self.dwtLevels-i-2][:,j,:,:].view(B,1,H,W), sigma, means=None, training=None) Berk

                self.ent_out_xo_list[(self.dwtLevels - i - 1) * 3 - j - 1].update_scale_table(self.scale_table)
                xo_indexes = self.ent_out_xo_list[(self.dwtLevels - i - 1) * 3 - j - 1].build_indexes(sigma)
                symbols_xo = self.ent_out_xo_list[(self.dwtLevels - i - 1) * 3 - j - 1].compress(
                    out_xo_list[self.dwtLevels - i - 2][:, j, :, :].view(B, 1, H, W), xo_indexes)
                self_informations_out_xo_list_dummy.append(symbols_xo[0])  # bitlengths
                xo_indexes_dummy.append(xo_indexes)
                byte_len_xo += len(symbols_xo[0])
                # out_xo_qnt = out_xo_qnt * self.scb_out_xo_list[i].expand_as(out_xo_qnt)  # backwards scale
                out_xo_list_qnt_dummy.append(out_xo_qnt)
                # con_channels = torch.cat((con_channels, out_xo_qnt), dim=1).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # update xo channels on which to condition
            xo_indexes_list.append(xo_indexes_dummy)
            self_informations_out_xo_list.append(self_informations_out_xo_list_dummy)
            con_channels = torch.cat((out_xo_list_qnt_dummy[0], out_xo_list_qnt_dummy[1], out_xo_list_qnt_dummy[2]),
                                     dim=1)
            out_xo_list_qnt.append(con_channels)
        # last operations before retruning
        out_xo_list_qnt.reverse()
        self_informations_out_xo_list.reverse()
        xo_indexes_list.reverse()
        # return all self_infos and qnt_tensors

        # decode
        shape = out_xe_qnt.shape
        deneme = self.decompress(self_informations_out_xe, self_informations_out_xo_list, shape)
        out_xo_dequant = []
        i = self.num_lifting_layers - 1

        out_xo_dequant.append(
            self.ent_out_xo_list[i * 3].decompress(self_informations_out_xo_list[3], xo_indexes_list[3]))
        out_xe_dequant = self.ent_out_xe.decompress(self_informations_out_xe, xe_indexes)

        for i in range(0, self.num_lifting_layers - 1, 1):
            out_xo_dequant_dummy = []
            for j in range(3):
                list_form = []
                list_form.append(self_informations_out_xo_list[self.dwtLevels - i - 2][j])
                out_xo_dequant_dummy.append(
                    self.ent_out_xo_list[(self.dwtLevels - i - 1) * 3 - j - 1].decompress(list_form, xo_indexes_list[
                        self.dwtLevels - i - 2][j]))
            out_xo_dequant.append(
                torch.cat((out_xo_dequant_dummy[0], out_xo_dequant_dummy[1], out_xo_dequant_dummy[2]), dim=1))

        out_xo_dequant.reverse()
        return self_informations_out_xe, self_informations_out_xo_list, out_xe_dequant, out_xo_dequant

    def decompress_ar(self, strings, shape, model_csc, param=None):
        kernel_size = 5
        padding = kernel_size//2
        y_hat = torch.zeros((shape[0],shape[1], shape[2]+padding, shape[3]+padding))
        if (shape[1] != 1):
            odd_even_flag = 1

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        decoder = RansDecoder()
        decoder.set_stream(strings[0])
        height = shape[2]
        width = shape[3]
        for h in range(height):
            for w in range(width):
                # y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size].cuda()
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                cscontext = self.csc_xe(y_crop)
                sigma = cscontext[:, 0:1, :, :]
                mu = cscontext[:, 1:2, :, :]
                sigma_val = sigma[:, :, padding:padding+1, padding:padding+1]
                indexes = self.ent_out_xe.build_indexes(sigma_val)
                list_form = []
                list_form.append(indexes.squeeze())
                rv = decoder.decode_stream(
                    list_form, cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.ent_out_xe.dequantize(rv)
                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp:hp+1, wp:wp+1] = rv
        decompressed = F.pad(-padding, -padding, - padding, -padding)
        print("done")
        return decompressed

    def compress_ar(self, y_hat, shape, network_kernel_size, plc, csc=None):
        padding = network_kernel_size//2
        B,C,height, width = shape[0], shape[1], shape[2], shape[3]
        # self.ent_out_xe.update_scale_table(self.scale_table)
        cdf = self.ent_out_xe.quantized_cdf.tolist()
        cdf_lengths = self.ent_out_xe.cdf_length.tolist()
        offsets = self.ent_out_xe.offset.tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + network_kernel_size, w: w + network_kernel_size]
                out_xe_qnt_mu_sigma = self.csc_xe.forward(y_crop)
                sigma = out_xe_qnt_mu_sigma[:, 0:1, :, :]
                mu = out_xe_qnt_mu_sigma[:, 1:2, :, :]
                sigma_val = sigma[:,:,padding, padding]
                indexes = self.ent_out_xe.build_indexes(sigma_val)
                y_crop = y_crop[:, :, padding, padding]
                y_q = self.ent_out_xe.quantize(y_crop, mode="symbols")
                y_hat[:, :, h + padding, w + padding] = y_q
                yq_list = []
                yq_list.append(y_q)
                index_list = []
                index_list.append(indexes)
                symbols_list.extend(yq_list)
                indexes_list.extend(index_list)

        print("done")
        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string
