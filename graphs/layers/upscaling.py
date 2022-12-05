import torch
import torch.nn as nn
import torch.functional as F

class upscale(nn.Module):
    def __init__(self):
        super(upscale, self).__init__()
        self.cnn_T_1 = nn.ConvTranspose2d(in_channels=1, out_channels=1,kernel_size=2, stride=2, padding = 0)
        self.cnn_1 = nn.Conv2d(1, 1, kernel_size=3, stride = 1 ,padding = 1,padding_mode = 'zeros')
        self.lrelu = torch.nn.LeakyReLU()
        self.resnet_coeff = 0.1
    def forward(self,input_image):
        up_sampled = self.cnn_T_1(input_image)
        up_sampled = self.lrelu(up_sampled)
        result = self.cnn_1(up_sampled)
        result = result + up_sampled * self.resnet_coeff
        return result

class zeroTreeWaveletPreviousLayer(nn.Module):
    def __init__(self, cfg,out_ch1):
        super(zeroTreeWaveletPreviousLayer, self).__init__()
        self.cfg = cfg
        self.convTranspose=nn.ModuleList()
        self.convBlocks=nn.ModuleList()
        self.convTranspose= upscale()
        self.convBlocks= (nn.Sequential(nn.Conv2d(1, out_ch1, kernel_size=1, stride=1, padding=1//2), nn.LeakyReLU(inplace=True)))

    def forward(self, high):
        high_upscaled = []
        B,C,W,H= high.shape

        high_upscaled = self.convBlocks(self.convTranspose(high))
        # high_upscaled_array = torch.cat((high_upscaled[0],high_upscaled[1], high_upscaled[2]), dim=1)
        return high_upscaled







