import torch
from torch import nn
import graphs.layers.basic_block as B


def init_weights(m):
    # print(m)


    if type(m) == nn.Conv2d:
        torch.nn.init.trunc_normal_(m.weight, std=0.01)
        # m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)
    # print(m.weight)



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
        recon = x - out
        return recon


class PostProcessResidual(nn.Module):
    def __init__(self, config):
        super(PostProcessResidual, self).__init__()
        self.channelNumber = config.clrch*64
        self.resWeight = 1
        self.resNet = nn.Sequential(
            nn.Conv2d(self.channelNumber, self.channelNumber, (3,3), stride=1, padding='same'), nn.ReLU(),  # GDN(iC * H),
            nn.Conv2d(self.channelNumber, self.channelNumber, (3,3), stride=1, padding='same')  # GDN(iC * H),
        )
        self.resNet.apply(init_weights)
    def forward(self, inputImage):
        tmp = self.resNet(inputImage)
        outputImage = tmp + inputImage
        return outputImage

class PostProcessingiWave(nn.Module):
    def __init__(self, config):
        super(PostProcessingiWave, self).__init__()
        self.channelNumber = config.clrch *64
        self.numberOfResnets = config.resnetlevel
        self.convFilter = nn.Conv2d(3, self.channelNumber, (3,3), stride=1, padding='same')
        self.interConvFilter = nn.Conv2d(self.channelNumber, self.channelNumber, (3, 3), stride=1, padding='same')
        self.outputConvFilter = nn.Conv2d(self.channelNumber, 3, (3,3), stride=1, padding='same')
        self.resNetList = nn.ModuleList()
        self.resWeight = 1
        for i in range(self.numberOfResnets):
            self.resNetList.append(PostProcessResidual(config))


    def forward(self, inputImage):
        tmp_1 = self.convFilter(inputImage)
        tmp_2 = tmp_1
        for i in range(self.numberOfResnets):
            tmp_2 = self.resNetList[i](tmp_2)
        tmp_2 = self.interConvFilter(tmp_2)
        tmp_2 = tmp_2 + tmp_1
        tmp_2 = self.outputConvFilter(tmp_2)
        reconImage = tmp_2 + inputImage
        return reconImage

class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L =[]
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = B.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x-n

import torch
import torch.nn as nn
import math


class _Residual_Block(nn.Module):
    def __init__(self,out_channels):
        super(_Residual_Block, self).__init__()

        #res1
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()
        #res1
        #concat1

        self.conv5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu6 = nn.PReLU()

        #res2
        self.conv7 = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()
        #res2
        #concat2

        self.conv9 = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu10 = nn.PReLU()

        #res3
        self.conv11 = nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()
        #res3

        self.conv13 = nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.up14 = nn.PixelShuffle(2)

        #concat2
        self.conv15 = nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*2, kernel_size=1, stride=1, padding=0, bias=False)
        #res4
        self.conv16 = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu17 = nn.PReLU()
        #res4

        self.conv18 = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.up19 = nn.PixelShuffle(2)

        #concat1
        self.conv20 = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        #res5
        self.conv21 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu24 = nn.PReLU()
        #res5

        self.conv25 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out

        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out

        out = self.relu10(self.conv9(out))
        res3 = out

        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)

        out = self.up14(self.conv13(out))

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)

        out = self.up19(self.conv18(out))

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out

class Recon_Block(nn.Module):
    def __init__(self,out_channels):
        super(Recon_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6= nn.PReLU()
        self.conv7 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)

        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)

        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)

        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output



class DIDN(nn.Module):
    def __init__(self,config):
        super(DIDN, self).__init__()
        self.out_channels = 64
        self.conv_input = nn.Conv2d(in_channels=config.clrch, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.conv_down = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = nn.PReLU()

        self.recursive_A = _Residual_Block(self.out_channels)
        self.recursive_B = _Residual_Block(self.out_channels)
        self.recursive_C = _Residual_Block(self.out_channels)
        self.recursive_D = _Residual_Block(self.out_channels)
        self.recursive_E = _Residual_Block(self.out_channels)
        self.recursive_F = _Residual_Block(self.out_channels)


        self.recon = Recon_Block(self.out_channels)
        #concat

        self.conv_mid = nn.Conv2d(in_channels=self.out_channels*6, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(in_channels=int(self.out_channels/4), out_channels=config.clrch, kernel_size=3, stride=1, padding=1, bias=False)




    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_input(x))
        out = self.relu2(self.conv_down(out))

        out1 = self.recursive_A(out)
        out2 = self.recursive_B(out1)
        out3 = self.recursive_C(out2)
        out4 = self.recursive_D(out3)
        out5 = self.recursive_E(out4)
        out6 = self.recursive_F(out5)

        recon1 = self.recon(out1)
        recon2 = self.recon(out2)
        recon3 = self.recon(out3)
        recon4 = self.recon(out4)
        recon5 = self.recon(out5)
        recon6 = self.recon(out6)

        out = torch.cat([recon1, recon2, recon3, recon4, recon5, recon6], 1)

        out = self.relu3(self.conv_mid(out))
        residual2 = out
        out = self.relu4(self.conv_mid2(out))
        out = torch.add(out, residual2)

        out= self.subpixel(out)
        out = self.conv_output(out)
        out = torch.add(out, residual)

        return out
from torch.nn import functional as F
class NNRegressor(nn.Module):
    def __init__(self):
        super(NNRegressor, self).__init__()
        self.mse = nn.MSELoss()

    def criterion(self, y, d):
        return self.mse(y, d)

class DnCNN(NNRegressor):

    def __init__(self, D, C=64):
        super(DnCNN, self).__init__()
        self.D = D

        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=1))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        for i in range(D):
            h = F.relu(self.bn[i](self.conv[i+1](h)))
        y = self.conv[D+1](h) + x
        return y


class UDnCNN(NNRegressor):

    def __init__(self, D, C=64):
        super(UDnCNN, self).__init__()
        self.D = D

        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=1))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        h_buff = []
        idx_buff = []
        shape_buff = []
        for i in range(D//2-1):
            shape_buff.append(h.shape)
            h, idx = F.max_pool2d(F.relu(self.bn[i](self.conv[i+1](h))),
                                  kernel_size=(2, 2), return_indices=True)
            h_buff.append(h)
            idx_buff.append(idx)
        for i in range(D//2-1, D//2+1):
            h = F.relu(self.bn[i](self.conv[i+1](h)))
        for i in range(D//2+1, D):
            j = i - (D // 2 + 1) + 1
            h = F.max_unpool2d(F.relu(self.bn[i](self.conv[i+1]((h+h_buff[-j])/np.sqrt(2)))),
                               idx_buff[-j], kernel_size=(2, 2), output_size=shape_buff[-j])
        y = self.conv[D+1](h) + x
        return y


class DUDnCNN(NNRegressor):

    def __init__(self, D, C=64):
        super(DUDnCNN, self).__init__()
        self.D = D

        # compute k(max_pool) and l(max_unpool)
        k = [0]
        k.extend([i for i in range(D//2)])
        k.extend([k[-1] for _ in range(D//2, D+1)])
        l = [0 for _ in range(D//2+1)]
        l.extend([i for i in range(D+1-(D//2+1))])
        l.append(l[-1])

        # holes and dilations for convolution layers
        holes = [2**(kl[0]-kl[1])-1 for kl in zip(k, l)]
        dilations = [i+1 for i in holes]

        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(
            nn.Conv2d(3, C, 3, padding=dilations[0], dilation=dilations[0]))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=dilations[i+1],
                                    dilation=dilations[i+1]) for i in range(D)])
        self.conv.append(
            nn.Conv2d(C, 3, 3, padding=dilations[-1], dilation=dilations[-1]))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        h_buff = []

        for i in range(D//2 - 1):
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1](h)
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))
            h_buff.append(h)

        for i in range(D//2 - 1, D//2 + 1):
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1](h)
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))

        for i in range(D//2 + 1, D):
            j = i - (D//2 + 1) + 1
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1]((h + h_buff[-j]) / np.sqrt(2))
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))

        y = self.conv[D+1](h) + x
        return y
